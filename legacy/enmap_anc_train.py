import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
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

class TransformerModel(nn.Module):
    def __init__(self, input_dim=73, d_model=73, nhead=73, num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # Reshape for Transformer
        x = x.unsqueeze(1).permute(1, 0, 2)  # Shape: (sequence_length, batch_size, d_model)
        x = self.pos_encoder(x)

        # Store intermediate features
        intermediate_features = []
        for layer in self.transformer_encoder.layers:
            x = layer(x)
            intermediate_features.append(x)  # Save intermediate features

        # Reshape back to (batch_size, sequence_length, d_model)
        x = x.permute(1, 0, 2)
        
        # Average over sequence length
        x_mean = x.mean(dim=1)  # Shape: (batch_size, d_model)
        
        # Predict salinity
        salinity_predictions = self.fc(x_mean)  # Shape: (batch_size, 1)
        salinity_predictions = self.sigmoid(salinity_predictions)  # Apply sigmoid to constrain output to [0, 1]

        return salinity_predictions, x_mean, intermediate_features  # Return intermediate features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Handle odd d_model
        if d_model % 2 == 1:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # Skip the last term if d_model is odd
        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe_resized = self.pe[:x.size(1), :].unsqueeze(0)
        x = x + pe_resized.expand(x.size(0), -1, -1)
        return x

# --- 2. Loss Functions ---
def task_loss_function(predictions, targets, huber_loss_criterion):
    """
    Compute task loss using model predictions.
    Args:
        predictions: Model predictions (shape: [batch_size, 1]).
        targets: Ground truth target values (shape: [batch_size]).
        huber_loss_criterion: Loss function for regression.
    """
    # Ensure shapes match
    targets = targets.unsqueeze(1)  # Reshape targets to [batch_size, 1]
    task_loss = huber_loss_criterion(predictions, targets)
    
    return task_loss

# --- 3. Data Loading and Preprocessing ---
def load_data(enmap_csv_path):
    return pd.read_csv(enmap_csv_path)

def preprocess_reflectance_and_locations(enmap_df):
    # Extract reflectance features
    enmap_bands = [f'Mean_Reflectance_Band_{i}' for i in range(1, 225)]
    excluded_bands = [f'Mean_Reflectance_Band_{i}' for i in range(130, 136)]
    enmap_reflectance_cols = [band for band in enmap_bands if band not in excluded_bands]
    enmap_reflectance = enmap_df[enmap_reflectance_cols].values
    
    # Extract 9 features
    feature_columns = [
        'chorizon.sandtotal_r', 
        'chorizon.silttotal_r', 
        'chorizon.claytotal_r', 
        'tm_min', 
        'tm_max', 
        'tm_avg', 
        'pr_min', 
        'pr_max', 
        'pr_avg'
    ]
    
    # Convert all feature columns to numeric, coercing errors to NaN
    for col in feature_columns:
        enmap_df[col] = pd.to_numeric(enmap_df[col], errors='coerce')
    
    features = enmap_df[feature_columns].values
    
    # Extract other data
    enmap_locations = enmap_df[['latitude', 'longitude']].values
    
    # Convert 'chorizon.ec_r' to numeric, handling non-numeric values
    enmap_salinity_ec = pd.to_numeric(enmap_df['chorizon.ec_r'], errors='coerce').values * 0.55
    
    enmap_aoi = enmap_df['AOI'].values
    
    # Filter out rows with missing values
    enmap_locations_df = pd.DataFrame(enmap_locations, columns=['latitude', 'longitude'])
    enmap_valid_location_indices = enmap_locations_df[~enmap_locations_df.isnull().any(axis=1)].index
    valid_indices_to_use = [i for i in enmap_valid_location_indices if i < enmap_reflectance.shape[0]]
    
    enmap_reflectance = enmap_reflectance[valid_indices_to_use]
    features = features[valid_indices_to_use]
    enmap_salinity_ec = enmap_salinity_ec[valid_indices_to_use]
    enmap_locations = enmap_locations[valid_indices_to_use]
    enmap_aoi = enmap_aoi[valid_indices_to_use]
    
    return enmap_reflectance, features, enmap_locations, enmap_salinity_ec, enmap_aoi, valid_indices_to_use

def split_data_indices_aoi(enmap_df, valid_indices_to_use=None, random_state=42):
    SEED = random_state
    data = enmap_df.iloc[valid_indices_to_use].copy()
    unique_aois = data['AOI'].unique()
    train_aois, test_valid_aois = train_test_split(unique_aois, test_size=0.4, random_state=SEED)
    test_aois, valid_aois = train_test_split(test_valid_aois, test_size=0.5, random_state=SEED)
    train_data_enmap = data[data['AOI'].isin(train_aois)]
    test_data_enmap = data[data['AOI'].isin(test_aois)]
    valid_data_enmap = data[data['AOI'].isin(valid_aois)]
    train_indices_aoi = train_data_enmap.reset_index(drop=True).index.tolist()
    val_indices_aoi = valid_data_enmap.reset_index(drop=True).index.tolist()
    test_indices_aoi = test_data_enmap.reset_index(drop=True).index.tolist()
    return train_indices_aoi, val_indices_aoi, test_indices_aoi, train_data_enmap, valid_data_enmap, test_data_enmap

def normalize_features(train_data, val_data, test_data):
    """
    Normalize features using statistics computed from the training set.
    Args:
        train_data: Training data (numpy array).
        val_data: Validation data (numpy array).
        test_data: Test data (numpy array).
    Returns:
        Normalized train, validation, and test data.
    """
    # Compute min and max from the training set
    data_min = np.nanmin(train_data, axis=0)  # Use np.nanmin to ignore NaN values
    data_max = np.nanmax(train_data, axis=0)  # Use np.nanmax to ignore NaN values
    data_range = data_max - data_min

    # Avoid division by zero by replacing zero ranges with 1
    data_range[data_range == 0] = 1

    # Normalize data
    train_data_normalized = (train_data - data_min) / data_range
    val_data_normalized = (val_data - data_min) / data_range
    test_data_normalized = (test_data - data_min) / data_range

    return train_data_normalized, val_data_normalized, test_data_normalized

def create_dataloaders(train_features, val_features, test_features, train_targets, val_targets, test_targets, batch_size=8, undersample_zeros=True):
    # Convert to tensors
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)

    if undersample_zeros:
        zero_indices = torch.where(train_targets_tensor == 0.0)[0]
        class_55_indices = torch.where(train_targets_tensor == 0.55)[0]
        minority_class_indices = torch.where(
            (train_targets_tensor != 0.0) & (train_targets_tensor != 0.55)
        )[0]
        num_zeros_to_keep = int(len(zero_indices) * 0.10)
        undersampled_zero_indices = zero_indices[torch.randperm(len(zero_indices))[:num_zeros_to_keep]]
        num_class_55_to_keep = int(len(class_55_indices) * 0.10)
        undersampled_class_55_indices = class_55_indices[torch.randperm(len(class_55_indices))[:num_class_55_to_keep]]
        balanced_train_indices = torch.cat([
            undersampled_zero_indices,
            undersampled_class_55_indices,
            minority_class_indices
        ])
        train_features_tensor = train_features_tensor[balanced_train_indices]
        train_targets_tensor = train_targets_tensor[balanced_train_indices]

    # Create datasets
    train_dataset = TensorDataset(train_features_tensor, train_targets_tensor)
    val_dataset = TensorDataset(torch.tensor(val_features, dtype=torch.float32), torch.tensor(val_targets, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32), torch.tensor(test_targets, dtype=torch.float32))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader

def generate_embeddings(reflectance, enmap_encoder_pretrained, device):
    """
    Generate embeddings for reflectance data using the pre-trained autoencoder.
    Args:
        reflectance: Reflectance data (numpy array).
        enmap_encoder_pretrained: Pre-trained autoencoder model.
        device: Device to use for computation (e.g., 'cuda' or 'cpu').
    Returns:
        embeddings: Encoded reflectance data (numpy array).
    """
    # Convert reflectance data to a tensor and move to the specified device
    reflectance_tensor = torch.tensor(reflectance, dtype=torch.float32).to(device)
    
    # Generate embeddings using the pre-trained autoencoder
    with torch.no_grad():
        _, embeddings = enmap_encoder_pretrained(reflectance_tensor)
        embeddings = embeddings.cpu().numpy()  # Move embeddings back to CPU and convert to numpy array
    
    return embeddings

# --- 4. Training and Validation Functions ---
def train_epoch(train_loader, model, optimizer, huber_loss_criterion, device):
    model.train()
    total_task_loss = 0

    for batch_idx, (features_batch, targets_batch) in enumerate(train_loader):
        features_batch = features_batch.to(device)
        targets_batch = targets_batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions, _, _ = model(features_batch)  # Use predictions instead of intermediate features

        # Debug: Check for NaNs in predictions and targets
        if torch.isnan(predictions).any() or torch.isnan(targets_batch).any():
            print("NaNs detected in predictions or targets!")
            sys.stdout.flush()

        # Compute task loss
        task_loss = task_loss_function(predictions, targets_batch, huber_loss_criterion)

        # Backward pass and optimization
        if not torch.isnan(task_loss):
            task_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_task_loss += task_loss.item()
        else:
            print("NaN loss detected, skipping this batch.")
            sys.stdout.flush()

    return total_task_loss / len(train_loader)

def validate_epoch(val_loader, model, huber_loss_criterion, device):
    model.eval()
    total_task_loss = 0

    with torch.no_grad():
        for batch_idx, (features_batch, targets_batch) in enumerate(val_loader):
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            # Forward pass
            predictions, _, _ = model(features_batch)  # Use predictions instead of intermediate features

            # Compute task loss
            task_loss = task_loss_function(predictions, targets_batch, huber_loss_criterion)

            total_task_loss += task_loss.item()

    return total_task_loss / len(val_loader)

def test_epoch(test_loader, model, huber_loss_criterion, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (features_batch, targets_batch) in enumerate(test_loader):
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            # Forward pass
            predictions, _, _ = model(features_batch)  # Use predictions instead of intermediate features

            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets_batch.cpu().numpy().flatten())

    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    return mae, r2, rmse

# --- 5. Main Function ---
def main():
    enmap_csv_path = '<DATA_ROOT>/multiplied_enmap_ssurgo_map.csv'
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64  # Reduced batch size for stability
    lr = 0.001  # Reduced learning rate
    weight_decay = 1e-5
    random_state = 42
    undersample_zeros = True

    print("Starting script for EnMAP training...")
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
    enmap_reflectance, features, enmap_locations, enmap_salinity_ec, enmap_aoi, valid_indices_to_use = preprocess_reflectance_and_locations(enmap_df)

    print("Shape of EnMAP reflectance data from CSV:", enmap_reflectance.shape)
    sys.stdout.flush()

    print("Splitting data indices by AOI...")
    sys.stdout.flush()
    train_indices_aoi, val_indices_aoi, test_indices_aoi, train_data_enmap, valid_data_enmap, test_data_enmap = split_data_indices_aoi(enmap_df, valid_indices_to_use, random_state)

    # Split data into train, validation, and test sets
    train_reflectance = enmap_reflectance[train_indices_aoi]
    val_reflectance = enmap_reflectance[val_indices_aoi]
    test_reflectance = enmap_reflectance[test_indices_aoi]

    train_features = features[train_indices_aoi]
    val_features = features[val_indices_aoi]
    test_features = features[test_indices_aoi]

    train_salinity_ec = enmap_salinity_ec[train_indices_aoi]
    val_salinity_ec = enmap_salinity_ec[val_indices_aoi]
    test_salinity_ec = enmap_salinity_ec[test_indices_aoi]

    # Normalize reflectance and 9 features independently
    train_reflectance, val_reflectance, test_reflectance = normalize_features(train_reflectance, val_reflectance, test_reflectance)
    train_features, val_features, test_features = normalize_features(train_features, val_features, test_features)

    # Load pre-trained autoencoder
    print("Loading pre-trained EnMAP encoder...")
    sys.stdout.flush()
    enmap_encoder_pretrained = SparseStackedAutoencoder(input_dim=train_reflectance.shape[1], latent_dim=64)
    enmap_encoder_pretrained.load_state_dict(torch.load('enmap_ssae.pth', map_location=device))
    enmap_encoder_pretrained.eval()
    enmap_encoder_pretrained.to(device)
    for param in enmap_encoder_pretrained.parameters():
        param.requires_grad = False

    # Generate embeddings for reflectance data
    print("Generating EnMAP embeddings...")
    sys.stdout.flush()
    train_embeddings = generate_embeddings(train_reflectance, enmap_encoder_pretrained, device)
    val_embeddings = generate_embeddings(val_reflectance, enmap_encoder_pretrained, device)
    test_embeddings = generate_embeddings(test_reflectance, enmap_encoder_pretrained, device)

    # Combine embeddings with 9 features
    print("Combining EnMAP embeddings and 9 features...")
    sys.stdout.flush()
    train_combined_features = np.hstack((train_embeddings, train_features))
    val_combined_features = np.hstack((val_embeddings, val_features))
    test_combined_features = np.hstack((test_embeddings, test_features))

    print("Shape of combined features (train):", train_combined_features.shape)
    print("Shape of combined features (val):", val_combined_features.shape)
    print("Shape of combined features (test):", test_combined_features.shape)
    sys.stdout.flush()

    # Drop rows with NaN values in combined features and targets
    print("Dropping rows with NaN values...")
    sys.stdout.flush()

    # Function to drop NaN rows
    def drop_nan_rows(features, targets):
        nan_mask = np.isnan(features).any(axis=1) | np.isnan(targets)
        return features[~nan_mask], targets[~nan_mask]

    # Drop NaN rows for train, validation, and test sets
    train_combined_features, train_salinity_ec = drop_nan_rows(train_combined_features, train_salinity_ec)
    val_combined_features, val_salinity_ec = drop_nan_rows(val_combined_features, val_salinity_ec)
    test_combined_features, test_salinity_ec = drop_nan_rows(test_combined_features, test_salinity_ec)

    print("Shape of combined features after dropping NaNs (train):", train_combined_features.shape)
    print("Shape of combined features after dropping NaNs (val):", val_combined_features.shape)
    print("Shape of combined features after dropping NaNs (test):", test_combined_features.shape)
    sys.stdout.flush()

    # Create DataLoaders
    print("Creating DataLoaders with AOI split and undersampling zeros...")
    sys.stdout.flush()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_combined_features, val_combined_features, test_combined_features,
        train_salinity_ec, val_salinity_ec, test_salinity_ec,
        batch_size=batch_size, undersample_zeros=undersample_zeros
    )

    # Initialize model and optimizer
    print("Initializing model and optimizer...")
    sys.stdout.flush()
    salinity_predictor = TransformerModel(input_dim=train_combined_features.shape[1]).to(device)
    optimizer = optim.Adam(salinity_predictor.parameters(), lr=lr, weight_decay=weight_decay)
    huber_loss_criterion = nn.HuberLoss()

    # Training and validation loop
    print("Starting training and validation...")
    sys.stdout.flush()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    early_stopping_patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(train_loader, salinity_predictor, optimizer, huber_loss_criterion, device)
        val_loss = validate_epoch(val_loader, salinity_predictor, huber_loss_criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        sys.stdout.flush()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(salinity_predictor.state_dict(), "best_enmap_salinity_predictor_combined.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                sys.stdout.flush()
                break

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("training_results_combined.png")
    plt.show()

    # Test set evaluation
    print("Starting test set evaluation...")
    sys.stdout.flush()
    mae, r2, rmse = test_epoch(test_loader, salinity_predictor, huber_loss_criterion, device)

    print(f"Test MAE: {mae:.4f}, Test R^2: {r2:.4f}, Test RMSE: {rmse:.4f}")
    sys.stdout.flush()

    # Save the model
    torch.save(salinity_predictor.state_dict(), "enmap_salinity_predictor_combined.pth")
    print("Saved EnMAP salinity predictor model.")
    sys.stdout.flush()
    
if __name__ == '__main__':
    main()