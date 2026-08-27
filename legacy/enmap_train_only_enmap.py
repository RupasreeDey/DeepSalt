import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import sys
import random

# Set environment for CUDA
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Print PyTorch and CUDA information
print(torch.__version__)
sys.stdout.flush()
print(torch.version.cuda)
sys.stdout.flush()
print(torch.cuda.is_available())
sys.stdout.flush()

# Disable memory-efficient SDP for stability
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# --- 1. Define Models ---
class SparseStackedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SparseStackedAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),  # Latent space (64-dimensional)
            nn.ReLU()
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),  # Reconstruct input
            nn.Sigmoid()  # Sigmoid to keep output in [0, 1] range
        )

    def forward(self, x):
        z = self.encoder(x)  # Latent representation
        x_recon = self.decoder(z)  # Reconstructed input
        return x_recon, z

# Define PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe_resized = self.pe[:x.size(1), :].unsqueeze(0)  # Resize PE to match input sequence length
        x = x + pe_resized.expand(x.size(0), -1, -1, )  # Expand PE to match batch size
        return x

# Define TransformerModel
class TransformerModel(nn.Module):
    def __init__(self, input_dim=64, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, x):
        # Ensure input tensor has shape (sequence_length, batch_size, input_dim)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = x.permute(1, 0, 2)  # Reshape to (sequence_length, batch_size, input_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # Pass through transformer
        x = x.permute(1, 0, 2)  # Reshape back to (batch_size, sequence_length, input_dim)
        x_mean = x.mean(dim=1)  # Average over sequence length
        salinity_predictions = self.fc(x_mean)
        salinity_predictions = self.sigmoid(salinity_predictions)  # Apply sigmoid to constrain output to [0, 1]
        salinity_predictions = 0.05 + (1.05 * salinity_predictions)  # Scale output to [0.05, 1.1]
        return salinity_predictions  # Return only predictions (no latent features)

# --- 2. Data Loading and Preprocessing ---
def load_data(enmap_csv_path):
    enmap_df = pd.read_csv(enmap_csv_path)
    return enmap_df

def preprocess_reflectance_and_locations(enmap_df):
    enmap_bands = [f'Mean_Band_{i}' for i in range(1, 225)]
    excluded_bands = [f'Mean_Band_{i}' for i in range(130, 136)]
    enmap_reflectance_cols = [band for band in enmap_bands if band not in excluded_bands]

    # Normalize EnMAP reflectance features to [0, 1]
    enmap_reflectance = enmap_df[enmap_reflectance_cols].values
    enmap_reflectance = (enmap_reflectance - enmap_reflectance.min()) / (enmap_reflectance.max() - enmap_reflectance.min())

    enmap_locations = enmap_df[['latitude', 'longitude']].values
    enmap_salinity_ec = enmap_df['chorizon.ec_r'].values * 0.55  # Multiply EnMAP EC values by 0.55
    enmap_aoi = enmap_df['AOI'].values

    # Filter out invalid locations
    enmap_locations_df = pd.DataFrame(enmap_locations, columns=['latitude', 'longitude'])
    enmap_valid_location_indices = enmap_locations_df[~enmap_locations_df.isnull().any(axis=1)].index
    valid_indices_to_use = [i for i in enmap_valid_location_indices if i < enmap_reflectance.shape[0]]

    enmap_reflectance = enmap_reflectance[valid_indices_to_use]
    enmap_salinity_ec = enmap_salinity_ec[valid_indices_to_use]
    enmap_locations = enmap_locations[valid_indices_to_use]
    enmap_aoi = enmap_aoi[valid_indices_to_use]

    return enmap_reflectance, enmap_locations, enmap_salinity_ec, enmap_aoi, valid_indices_to_use

def scale_salinity_ec(enmap_salinity_ec_np):
    """Skip scaling and return the original EnMAP salinity/EC values."""
    enmap_salinity_ec_scaled_tensor = torch.tensor(enmap_salinity_ec_np, dtype=torch.float32)
    return enmap_salinity_ec_scaled_tensor, None

def generate_embeddings(enmap_reflectance, enmap_encoder_pretrained, device):
    """Generates embeddings for EnMAP reflectance data using pretrained encoders."""
    enmap_reflectance_tensor = torch.tensor(enmap_reflectance, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, enmap_embeddings = enmap_encoder_pretrained(enmap_reflectance_tensor)  # Extract latent representation (z)
        enmap_embeddings = enmap_embeddings.cpu().numpy()  # Move to CPU and convert to numpy
    return enmap_embeddings

def split_data_indices_aoi(enmap_df, valid_indices_to_use=None, random_state=42):
    """Splits EnMAP data indices into train, validation, and test sets based on AOI."""
    SEED = random_state
    data = enmap_df.iloc[valid_indices_to_use].copy()
    unique_aois = data['AOI'].unique()
    train_aois, test_valid_aois = train_test_split(unique_aois, test_size=0.2, random_state=SEED)
    test_aois, valid_aois = train_test_split(test_valid_aois, test_size=0.5, random_state=SEED)

    train_data_enmap = data[data['AOI'].isin(train_aois)]
    test_data_enmap = data[data['AOI'].isin(test_aois)]
    valid_data_enmap = data[data['AOI'].isin(valid_aois)]

    train_indices_aoi = train_data_enmap.reset_index(drop=True).index.tolist()
    val_indices_aoi = valid_data_enmap.reset_index(drop=True).index.tolist()
    test_indices_aoi = test_data_enmap.reset_index(drop=True).index.tolist()

    return train_indices_aoi, val_indices_aoi, test_indices_aoi, train_data_enmap, valid_data_enmap, test_data_enmap

def create_dataloaders(enmap_embeddings_np, enmap_salinity_ec_scaled_tensor, train_indices, val_indices, test_indices, batch_size=8, undersample_zeros=True):
    """Creates DataLoaders for training, validation, and testing with undersampling."""
    train_enmap_embeddings_tensor = torch.tensor(enmap_embeddings_np[train_indices], dtype=torch.float32)
    train_enmap_salinity_ec_tensor = enmap_salinity_ec_scaled_tensor[train_indices]

    if undersample_zeros:
        # Identify indices for majority classes (0.0 and 0.55)
        zero_indices = torch.where(train_enmap_salinity_ec_tensor == 0.0)[0]
        class_55_indices = torch.where(train_enmap_salinity_ec_tensor == 0.55)[0]
        minority_class_indices = torch.where(
            (train_enmap_salinity_ec_tensor != 0.0) & (train_enmap_salinity_ec_tensor != 0.55)
        )[0]

        # Undersample majority classes
        num_zeros_to_keep = int(len(zero_indices) * 0.10)
        undersampled_zero_indices = zero_indices[torch.randperm(len(zero_indices))[:num_zeros_to_keep]]

        num_class_55_to_keep = int(len(class_55_indices) * 0.10)
        undersampled_class_55_indices = class_55_indices[torch.randperm(len(class_55_indices))[:num_class_55_to_keep]]

        # Combine undersampled majority indices with minority indices
        balanced_train_indices = torch.cat([
            undersampled_zero_indices,
            undersampled_class_55_indices,
            minority_class_indices
        ])
        train_indices_for_loader = [train_indices[i] for i in balanced_train_indices.tolist()]
    else:
        train_indices_for_loader = train_indices

    # Create TensorDataset and DataLoader for training
    train_enmap_embeddings_tensor_balanced = torch.tensor(enmap_embeddings_np[train_indices_for_loader], dtype=torch.float32)
    train_enmap_salinity_ec_tensor_balanced = enmap_salinity_ec_scaled_tensor[train_indices_for_loader]
    train_dataset = TensorDataset(train_enmap_embeddings_tensor_balanced, train_enmap_salinity_ec_tensor_balanced)
    train_loader_paired = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Create TensorDataset and DataLoader for validation
    val_enmap_embeddings_tensor = torch.tensor(enmap_embeddings_np[val_indices], dtype=torch.float32)
    val_enmap_salinity_ec_tensor = enmap_salinity_ec_scaled_tensor[val_indices]
    val_dataset = TensorDataset(val_enmap_embeddings_tensor, val_enmap_salinity_ec_tensor)
    val_loader_paired = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Create TensorDataset and DataLoader for testing
    test_enmap_embeddings_tensor = torch.tensor(enmap_embeddings_np[test_indices], dtype=torch.float32)
    test_enmap_salinity_ec_tensor = enmap_salinity_ec_scaled_tensor[test_indices]
    test_dataset = TensorDataset(test_enmap_embeddings_tensor, test_enmap_salinity_ec_tensor)
    test_loader_paired = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader_paired, val_loader_paired, test_loader_paired

# --- 3. Training and Validation ---
def train_epoch(train_loader_paired, enmap_salinity_predictor, optimizer, criterion, device):
    """Trains the EnMAP salinity predictor model for one epoch."""
    enmap_salinity_predictor.train()
    train_loss = 0

    for batch_idx, (enmap_embeddings_batch, enmap_salinity_ec_batch) in enumerate(train_loader_paired):
        enmap_embeddings_batch = enmap_embeddings_batch.to(device)
        enmap_salinity_ec_batch = enmap_salinity_ec_batch.to(device)

        optimizer.zero_grad()

        # Student model predictions
        salinity_predictions = enmap_salinity_predictor(enmap_embeddings_batch)

        # Compute loss
        loss = criterion(salinity_predictions.squeeze(), enmap_salinity_ec_batch)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Accumulate loss
        train_loss += loss.item()

    num_batches = len(train_loader_paired)
    train_loss /= num_batches

    return train_loss

def validate_epoch(val_loader_paired, enmap_salinity_predictor, criterion, device):
    """Validates the EnMAP salinity predictor model for one epoch."""
    enmap_salinity_predictor.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, (enmap_embeddings_batch, enmap_salinity_ec_batch) in enumerate(val_loader_paired):
            enmap_embeddings_batch = enmap_embeddings_batch.to(device)
            enmap_salinity_ec_batch = enmap_salinity_ec_batch.to(device)

            # Student model predictions
            salinity_predictions = enmap_salinity_predictor(enmap_embeddings_batch)

            # Compute loss
            loss = criterion(salinity_predictions.squeeze(), enmap_salinity_ec_batch)

            # Accumulate loss
            val_loss += loss.item()

    num_batches = len(val_loader_paired)
    val_loss /= num_batches

    return val_loss

def test_epoch(test_loader_paired, enmap_salinity_predictor, criterion, device):
    """Evaluates the EnMAP salinity predictor model on the test set."""
    enmap_salinity_predictor.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (enmap_embeddings_batch, enmap_salinity_ec_batch) in enumerate(test_loader_paired):
            enmap_embeddings_batch = enmap_embeddings_batch.to(device)
            enmap_salinity_ec_batch = enmap_salinity_ec_batch.to(device)

            # Student model predictions
            salinity_predictions_batch = enmap_salinity_predictor(enmap_embeddings_batch)

            # Compute loss
            loss = criterion(salinity_predictions_batch.squeeze(), enmap_salinity_ec_batch)

            # Accumulate loss
            test_loss += loss.item()

            # Store predictions and targets for MAE calculation
            all_predictions.extend(salinity_predictions_batch.cpu().numpy().flatten())
            all_targets.extend(enmap_salinity_ec_batch.cpu().numpy().flatten())

    num_batches = len(test_loader_paired)
    test_loss /= num_batches
    mae = mean_absolute_error(all_targets, all_predictions)

    return test_loss, mae

# --- 4. Main Function ---
def main():
    enmap_csv_path = '<DATA_ROOT>/valid_enmap_reflectance_with_ec_mask.csv'
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    lr = 0.001
    weight_decay = 1e-5
    random_state = 42
    undersample_zeros = True

    print("Starting script for training student model without KD...")
    sys.stdout.flush()

    print(f"CUDA available: {torch.cuda.is_available()}")
    sys.stdout.flush()
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        sys.stdout.flush()

    print("Loading data from CSV...")
    sys.stdout.flush()
    enmap_df = pd.read_csv(enmap_csv_path)

    # Preprocess data
    enmap_reflectance, enmap_locations, enmap_salinity_ec, enmap_aoi, valid_indices_to_use = preprocess_reflectance_and_locations(enmap_df)

    print("Shape of EnMAP reflectance data from CSV:", enmap_reflectance.shape)
    sys.stdout.flush()

    print("Loading pre-trained EnMAP encoder...")
    sys.stdout.flush()
    enmap_encoder_pretrained = SparseStackedAutoencoder(input_dim=enmap_reflectance.shape[1], latent_dim=64)
    enmap_encoder_pretrained.load_state_dict(torch.load('enmap_ssae.pth', map_location=device))
    enmap_encoder_pretrained.eval()
    enmap_encoder_pretrained.to(device)
    for param in enmap_encoder_pretrained.parameters():
        param.requires_grad = False

    print("Generating EnMAP embeddings...")
    sys.stdout.flush()
    enmap_embeddings = generate_embeddings(enmap_reflectance, enmap_encoder_pretrained, device)
    print("Shape of EnMAP embeddings:", enmap_embeddings.shape)
    sys.stdout.flush()

    print("Scaling EnMAP salinity/EC...")
    sys.stdout.flush()
    enmap_salinity_ec_scaled_tensor, _ = scale_salinity_ec(enmap_salinity_ec)

    print("Splitting data indices by AOI...")
    sys.stdout.flush()
    train_indices_aoi, val_indices_aoi, test_indices_aoi, train_data_enmap, valid_data_enmap, test_data_enmap = split_data_indices_aoi(enmap_df, valid_indices_to_use, random_state)

    print("Creating DataLoaders with AOI split and undersampling zeros...")
    sys.stdout.flush()
    train_loader_paired, val_loader_paired, test_loader_paired = create_dataloaders(enmap_embeddings, enmap_salinity_ec_scaled_tensor, train_indices_aoi, val_indices_aoi, test_indices_aoi, batch_size, undersample_zeros=undersample_zeros)

    print("Initializing student model and optimizer...")
    sys.stdout.flush()
    # Initialize student model (EnMAP salinity predictor)
    enmap_salinity_predictor = TransformerModel(input_dim=64).to(device)

    # Initialize optimizer
    optimizer = optim.Adam(enmap_salinity_predictor.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize loss function
    criterion = nn.SmoothL1Loss()  # Use Smooth L1 Loss for regression

    print("Starting training and validation...")
    sys.stdout.flush()
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_loss = train_epoch(train_loader_paired, enmap_salinity_predictor, optimizer, criterion, device)
        val_loss = validate_epoch(val_loader_paired, enmap_salinity_predictor, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        sys.stdout.flush()

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("training_results_enmap_only_model.png")
    plt.show()

     # --- Test Set Evaluation ---
    print("Starting test set evaluation...")
    sys.stdout.flush()

    # Evaluate on the test set
    test_loss, test_mae = test_epoch(test_loader_paired, enmap_salinity_predictor, criterion, device)

    # Print test set metrics
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    sys.stdout.flush()

    # Save the trained EnMAP salinity predictor model
    torch.save(enmap_salinity_predictor.state_dict(), "enmap_salinity_predictor_kd.pth")
    print("Saved EnMAP salinity predictor model.")
    sys.stdout.flush()

if __name__ == '__main__':
    main()