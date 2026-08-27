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
from sklearn.decomposition import PCA
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

# --- 1. Define Transformer Encoder and Salinity Predictor Models ---
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
        # Encode input to latent space
        z = self.encoder(x)
        # Decode latent space to reconstruct input
        x_recon = self.decoder(z)
        return x_recon, z

# Define TransformerModel with constrained output
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
        x = x.mean(dim=1)  # Average over sequence length
        x = self.fc(x)
        x = self.sigmoid(x)  # Apply sigmoid to constrain output to [0, 1]
        x = 0.05 + (1.05 * x)  # Scale output to [0.05, 1.1]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe_resized = self.pe[:x.size(1), :].unsqueeze(0)  # Resize PE to match input sequence length
        x = x + pe_resized.expand(x.size(0), -1, -1)  # Expand PE to match batch size
        return x

# --- 2. Loss Functions ---
def temperature_scaled_mse_loss(student_preds, teacher_preds, temperature=1.0):
    """Compute temperature-scaled MSE loss for regression."""
    # Clip teacher predictions to the EnMAP data scale [0, 1.1]
    teacher_preds_clipped = torch.clamp(teacher_preds, min=0.0, max=1.1)

    student_preds_scaled = student_preds / temperature
    teacher_preds_scaled = teacher_preds_clipped / temperature
    return F.mse_loss(student_preds_scaled, teacher_preds_scaled)

def knowledge_distillation_loss(student_preds, teacher_preds, targets, temperature=1.0, alpha=0.4, smooth_l1_loss_criterion=None):
    """Compute combined loss with temperature-scaled MSE and task loss."""
    # Clip teacher predictions to the EnMAP data scale [0, 1.1]
    teacher_preds_clipped = torch.clamp(teacher_preds, min=0.0, max=1.1)

    # Apply class weights
    weights = torch.where(targets > 0, torch.tensor(10.0).to(targets.device), torch.tensor(1.0).to(targets.device))
    task_loss = smooth_l1_loss_criterion(student_preds, targets.unsqueeze(1))
    task_loss = (task_loss * weights).mean()  # Weighted task loss

    # Temperature-scaled MSE loss for distillation
    distillation_loss = temperature_scaled_mse_loss(student_preds, teacher_preds_clipped, temperature)

    return alpha * distillation_loss + (1 - alpha) * task_loss

# --- 3. Data Loading and Preprocessing ---
def remove_outliers_iqr(data, column):
    """
    Remove outliers from a column in the dataset using the IQR method.
    
    Args:
        data (pd.DataFrame): The dataset.
        column (str): The column to remove outliers from.
    
    Returns:
        pd.DataFrame: The dataset with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out outliers
    data_cleaned = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data_cleaned

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

    # Print embeddings for the first few samples
    print("EnMAP Embeddings (First 5 samples):")
    print(enmap_embeddings[:5])
    sys.stdout.flush()

    return enmap_embeddings

def split_data_indices_aoi(enmap_df, valid_indices_to_use=None, random_state=42):
    """Splits EnMAP data indices into train, validation, and test sets based on AOI."""
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

def create_dataloaders(enmap_embeddings_np, enmap_salinity_ec_scaled_tensor, train_indices, val_indices, test_indices, batch_size=8, undersample_zeros=True):
    """Creates DataLoaders for training, validation, and testing with undersampling."""
    train_enmap_embeddings_tensor = torch.tensor(enmap_embeddings_np[train_indices], dtype=torch.float32)
    train_enmap_salinity_ec_tensor = enmap_salinity_ec_scaled_tensor[train_indices]

    # Print counts of distinct target values before undersampling
    distinct_values, value_counts = torch.unique(train_enmap_salinity_ec_tensor, return_counts=True)
    print("Counts of distinct target values before undersampling:")
    for value, count in zip(distinct_values, value_counts):
        print(f"Value: {value.item()}, Count: {count.item()}")
        sys.stdout.flush()

    if undersample_zeros:
        # Identify indices for majority classes (0.0 and 0.55)
        zero_indices = torch.where(train_enmap_salinity_ec_tensor == 0.0)[0]
        class_55_indices = torch.where(train_enmap_salinity_ec_tensor == 0.55)[0]
        minority_class_indices = torch.where(
            (train_enmap_salinity_ec_tensor != 0.0) & (train_enmap_salinity_ec_tensor != 0.55)
        )[0]

        # Undersample majority classes
        # Take 10% of zeros
        num_zeros_to_keep = int(len(zero_indices) * 0.10)
        undersampled_zero_indices = zero_indices[torch.randperm(len(zero_indices))[:num_zeros_to_keep]]

        # Take 10% of 0.55
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

# --- 4. Training and Validation ---
def compute_non_zero_metrics(predictions, targets):
    """Compute metrics (MAE, loss) for non-zero salinity samples."""
    non_zero_mask = targets > 0
    if non_zero_mask.sum() > 0:  # Check if there are non-zero samples
        non_zero_predictions = predictions[non_zero_mask]
        non_zero_targets = targets[non_zero_mask]
        non_zero_mae = mean_absolute_error(non_zero_targets.cpu().detach().numpy(), non_zero_predictions.cpu().detach().numpy())
        non_zero_loss = F.smooth_l1_loss(non_zero_predictions, non_zero_targets.unsqueeze(1)).item()
    else:
        non_zero_mae = 0.0
        non_zero_loss = 0.0
    return non_zero_mae, non_zero_loss

def train_epoch(train_loader_paired, enmap_salinity_predictor, pretrained_ftir_salinity_predictor, optimizer_enmap, temperature_scaled_mse_loss, knowledge_distillation_loss, smooth_l1_loss_criterion, epoch, num_epochs, device, temperature, alpha):
    """Trains the EnMAP salinity predictor model for one epoch."""
    enmap_salinity_predictor.train()
    pretrained_ftir_salinity_predictor.eval()

    train_teacher_loss_epoch = 0
    train_task_loss_epoch = 0
    train_combined_loss_epoch = 0
    train_salinity_mae_epoch = 0
    train_non_zero_mae_epoch = 0
    train_non_zero_loss_epoch = 0

    all_student_preds = []
    all_teacher_preds = []
    all_enmap_ec_batches = []

    for batch_idx, (enmap_embeddings_batch, enmap_salinity_ec_batch) in enumerate(train_loader_paired):
        enmap_embeddings_batch = enmap_embeddings_batch.to(device)
        enmap_salinity_ec_batch = enmap_salinity_ec_batch.to(device)

        optimizer_enmap.zero_grad()

        # Student model predictions
        student_salinity_predictions = enmap_salinity_predictor(enmap_embeddings_batch)

        # Teacher model predictions
        with torch.no_grad():
            teacher_salinity_predictions = pretrained_ftir_salinity_predictor(enmap_embeddings_batch)

        # Compute losses
        teacher_loss = temperature_scaled_mse_loss(student_salinity_predictions, teacher_salinity_predictions, temperature)
        task_loss = smooth_l1_loss_criterion(student_salinity_predictions, enmap_salinity_ec_batch.unsqueeze(1))
        combined_loss = knowledge_distillation_loss(student_salinity_predictions, teacher_salinity_predictions, enmap_salinity_ec_batch, temperature, alpha, smooth_l1_loss_criterion)

        # Compute MAE
        salinity_mae = mean_absolute_error(
            enmap_salinity_ec_batch.cpu().detach().numpy(),
            student_salinity_predictions.cpu().detach().numpy()
        )

        # Compute non-zero metrics
        non_zero_mae, non_zero_loss = compute_non_zero_metrics(student_salinity_predictions, enmap_salinity_ec_batch)

        # Backpropagation
        combined_loss.backward()
        optimizer_enmap.step()

        # Accumulate losses
        train_teacher_loss_epoch += teacher_loss.item()
        train_task_loss_epoch += task_loss.item()
        train_combined_loss_epoch += combined_loss.item()
        train_salinity_mae_epoch += salinity_mae
        train_non_zero_mae_epoch += non_zero_mae
        train_non_zero_loss_epoch += non_zero_loss

        # Store predictions and actual values for printing in specific epochs
        all_student_preds.append(student_salinity_predictions.cpu().detach().numpy())
        all_teacher_preds.append(teacher_salinity_predictions.cpu().numpy())
        all_enmap_ec_batches.append(enmap_salinity_ec_batch.cpu().numpy())

    num_batches = len(train_loader_paired)
    train_teacher_loss_epoch /= num_batches
    train_task_loss_epoch /= num_batches
    train_combined_loss_epoch /= num_batches
    train_salinity_mae_epoch /= num_batches
    train_non_zero_mae_epoch /= num_batches
    train_non_zero_loss_epoch /= num_batches

    return train_teacher_loss_epoch, train_task_loss_epoch, train_combined_loss_epoch, train_salinity_mae_epoch, train_non_zero_mae_epoch, train_non_zero_loss_epoch, all_student_preds, all_teacher_preds, all_enmap_ec_batches

def validate_epoch(val_loader_paired, enmap_salinity_predictor, pretrained_ftir_salinity_predictor, temperature_scaled_mse_loss, knowledge_distillation_loss, smooth_l1_loss_criterion, device, temperature, alpha):
    """Validates the EnMAP salinity predictor model for one epoch."""
    enmap_salinity_predictor.eval()
    pretrained_ftir_salinity_predictor.eval()

    val_teacher_loss_epoch = 0
    val_task_loss_epoch = 0
    val_combined_loss_epoch = 0
    val_salinity_mae_epoch = 0
    val_non_zero_mae_epoch = 0
    val_non_zero_loss_epoch = 0

    all_student_preds_val = []
    all_teacher_preds_val = []
    all_enmap_ec_batches_val = []

    with torch.no_grad():
        for batch_idx, (enmap_embeddings_batch_val, enmap_salinity_ec_batch_val) in enumerate(val_loader_paired):
            enmap_embeddings_batch_val = enmap_embeddings_batch_val.to(device)
            enmap_salinity_ec_batch_val = enmap_salinity_ec_batch_val.to(device)

            # Student model predictions
            student_salinity_predictions_val = enmap_salinity_predictor(enmap_embeddings_batch_val)

            # Teacher model predictions
            teacher_salinity_predictions_val = pretrained_ftir_salinity_predictor(enmap_embeddings_batch_val)

            # Compute losses
            val_teacher_loss = temperature_scaled_mse_loss(student_salinity_predictions_val, teacher_salinity_predictions_val, temperature)
            val_task_loss = smooth_l1_loss_criterion(student_salinity_predictions_val, enmap_salinity_ec_batch_val.unsqueeze(1))
            val_combined_loss = knowledge_distillation_loss(student_salinity_predictions_val, teacher_salinity_predictions_val, enmap_salinity_ec_batch_val, temperature, alpha, smooth_l1_loss_criterion)

            # Compute MAE
            val_salinity_mae = mean_absolute_error(
                enmap_salinity_ec_batch_val.cpu().detach().numpy(),
                student_salinity_predictions_val.cpu().detach().numpy()
            )

            # Compute non-zero metrics
            non_zero_mae, non_zero_loss = compute_non_zero_metrics(student_salinity_predictions_val, enmap_salinity_ec_batch_val)

            # Accumulate losses
            val_teacher_loss_epoch += val_teacher_loss.item()
            val_task_loss_epoch += val_task_loss.item()
            val_combined_loss_epoch += val_combined_loss.item()
            val_salinity_mae_epoch += val_salinity_mae
            val_non_zero_mae_epoch += non_zero_mae
            val_non_zero_loss_epoch += non_zero_loss

            # Store predictions and actual values for printing in specific epochs
            all_student_preds_val.append(student_salinity_predictions_val.cpu().numpy())
            all_teacher_preds_val.append(teacher_salinity_predictions_val.cpu().numpy())
            all_enmap_ec_batches_val.append(enmap_salinity_ec_batch_val.cpu().numpy())

    num_batches = len(val_loader_paired)
    val_teacher_loss_epoch /= num_batches
    val_task_loss_epoch /= num_batches
    val_combined_loss_epoch /= num_batches
    val_salinity_mae_epoch /= num_batches
    val_non_zero_mae_epoch /= num_batches
    val_non_zero_loss_epoch /= num_batches

    return val_teacher_loss_epoch, val_task_loss_epoch, val_combined_loss_epoch, val_salinity_mae_epoch, val_non_zero_mae_epoch, val_non_zero_loss_epoch, all_student_preds_val, all_teacher_preds_val, all_enmap_ec_batches_val

def train_and_validate(num_epochs, train_loader_paired, val_loader_paired, enmap_salinity_predictor, pretrained_ftir_salinity_predictor, optimizer_enmap, temperature_scaled_mse_loss, knowledge_distillation_loss, smooth_l1_loss_criterion, device, temperature, alpha, scheduler):
    """Trains and validates the EnMAP salinity predictor model over a specified number of epochs."""
    train_teacher_losses = []
    val_teacher_losses = []
    train_task_losses = []
    val_task_losses = []
    train_combined_losses = []
    val_combined_losses = []
    train_salinity_mae_losses = []
    val_salinity_mae_losses = []
    train_non_zero_mae_losses = []
    val_non_zero_mae_losses = []
    train_non_zero_loss_losses = []
    val_non_zero_loss_losses = []

    best_val_loss = float('inf')
    patience = 100
    counter = 0

    print("Starting Training of EnMAP Salinity Predictor with KD and AOI Split...")
    sys.stdout.flush()
    for epoch in range(num_epochs):
        train_teacher_loss_epoch, train_task_loss_epoch, train_combined_loss_epoch, train_salinity_mae_epoch, train_non_zero_mae_epoch, train_non_zero_loss_epoch, all_student_preds, all_teacher_preds, all_enmap_ec_batches = train_epoch(
            train_loader_paired, enmap_salinity_predictor, pretrained_ftir_salinity_predictor, optimizer_enmap, temperature_scaled_mse_loss, knowledge_distillation_loss, smooth_l1_loss_criterion, epoch, num_epochs, device, temperature, alpha
        )
        val_teacher_loss_epoch, val_task_loss_epoch, val_combined_loss_epoch, val_salinity_mae_epoch, val_non_zero_mae_epoch, val_non_zero_loss_epoch, all_student_preds_val, all_teacher_preds_val, all_enmap_ec_batches_val = validate_epoch(
            val_loader_paired, enmap_salinity_predictor, pretrained_ftir_salinity_predictor, temperature_scaled_mse_loss, knowledge_distillation_loss, smooth_l1_loss_criterion, device, temperature, alpha
        )

        train_teacher_losses.append(train_teacher_loss_epoch)
        train_task_losses.append(train_task_loss_epoch)
        train_combined_losses.append(train_combined_loss_epoch)
        train_salinity_mae_losses.append(train_salinity_mae_epoch)
        train_non_zero_mae_losses.append(train_non_zero_mae_epoch)
        train_non_zero_loss_losses.append(train_non_zero_loss_epoch)

        val_teacher_losses.append(val_teacher_loss_epoch)
        val_task_losses.append(val_task_loss_epoch)
        val_combined_losses.append(val_combined_loss_epoch)
        val_salinity_mae_losses.append(val_salinity_mae_epoch)
        val_non_zero_mae_losses.append(val_non_zero_mae_epoch)
        val_non_zero_loss_losses.append(val_non_zero_loss_epoch)

        # Update learning rate scheduler
        scheduler.step(val_combined_loss_epoch)

        # Early stopping logic
        if val_combined_loss_epoch < best_val_loss:
            best_val_loss = val_combined_loss_epoch
            counter = 0
            # Save the best model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': enmap_salinity_predictor.state_dict(),
                'optimizer_state_dict': optimizer_enmap.state_dict(),
                'loss': val_combined_loss_epoch,
            }, 'enmap_salinity_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Combined Loss: {train_combined_losses[-1]:.4f}, Val Combined Loss: {val_combined_losses[-1]:.4f}, Train Teacher KLD Loss: {train_teacher_losses[-1]:.4f}, Val Teacher KLD Loss: {val_teacher_losses[-1]:.4f}, Train Task SmoothL1 Loss: {train_task_losses[-1]:.4f}, Val Task SmoothL1 Loss: {val_task_losses[-1]:.4f}, Train Salinity MAE: {train_salinity_mae_losses[-1]:.4f}, Val Salinity MAE: {val_salinity_mae_losses[-1]:.4f}, Train Non-Zero MAE: {train_non_zero_mae_losses[-1]:.4f}, Val Non-Zero MAE: {val_non_zero_mae_losses[-1]:.4f}, Train Non-Zero Loss: {train_non_zero_loss_losses[-1]:.4f}, Val Non-Zero Loss: {val_non_zero_loss_losses[-1]:.4f}')
        sys.stdout.flush()

        if (epoch + 1) % 5 == 0:
            print(f"--- Epoch {epoch + 1} Predictions (First Batch from Training Set) ---")
            sys.stdout.flush()
            print("Student Predictions (Train Batch 1):", all_student_preds[0][:5].flatten()) # Print first 5 predictions from the first batch
            sys.stdout.flush()
            print("Teacher Predictions (Train Batch 1):", all_teacher_preds[0][:5].flatten()) # Print first 5 predictions from the first batch
            sys.stdout.flush()
            print("Actual EnMAP EC (Train Batch 1):", all_enmap_ec_batches[0][:5].flatten()) # Print first 5 actual values from the first batch
            sys.stdout.flush()

            print(f"--- Epoch {epoch + 1} Predictions (First Batch from Validation Set) ---")
            sys.stdout.flush()
            print("Student Predictions (Validation Batch 1):", all_student_preds_val[0][:5].flatten()) # Print first 5 predictions from the first batch
            sys.stdout.flush()
            print("Teacher Predictions (Validation Batch 1):", all_teacher_preds_val[0][:5].flatten()) # Print first 5 predictions from the first batch
            sys.stdout.flush()
            print("Actual EnMAP EC (Validation Batch 1):", all_enmap_ec_batches_val[0][:5].flatten()) # Print first 5 actual values from the first batch
            sys.stdout.flush()

    return train_teacher_losses, val_teacher_losses, train_task_losses, val_task_losses, train_combined_losses, val_combined_losses, train_salinity_mae_losses, val_salinity_mae_losses, train_non_zero_mae_losses, val_non_zero_mae_losses, train_non_zero_loss_losses, val_non_zero_loss_losses

def initialize_models_and_optimizer(enmap_reflectance_shape, device, lr=0.001, weight_decay=1e-5):
    """
    Initializes pretrained FTIR salinity predictor, EnMAP encoder, EnMAP salinity predictor,
    optimizer, loss criterion, and learning rate scheduler.
    """
    # Load pretrained FTIR salinity predictor (teacher model)
    pretrained_ftir_salinity_predictor = TransformerModel(input_dim=64)  # Ensure input_dim matches EnMAP embeddings

    # Load checkpoint and inspect
    checkpoint = torch.load('ftir_salinity_model_ssae_clipped.pth', map_location=device)
    print("Checkpoint Keys:", checkpoint.keys())
    print("State Dict Shape:", {k: v.shape for k, v in checkpoint.items()})

    # Load state dict into model
    pretrained_ftir_salinity_predictor.load_state_dict(checkpoint)
    pretrained_ftir_salinity_predictor.eval()
    pretrained_ftir_salinity_predictor.to(device)

    # Freeze parameters
    for param in pretrained_ftir_salinity_predictor.parameters():
        param.requires_grad = False

    # Print model architecture
    print("Teacher Model Architecture:")
    print(pretrained_ftir_salinity_predictor)

    # Test teacher model on random inputs
    random_input = torch.randn(5, 64).to(device)  # Random input with the same shape as embeddings
    with torch.no_grad():
        random_predictions = pretrained_ftir_salinity_predictor(random_input)
        print("Teacher Predictions on Random Input:", random_predictions.cpu().numpy())

    # Inspect final layer weights
    for name, param in pretrained_ftir_salinity_predictor.named_parameters():
        if "fc" in name:  # Assuming the final layer is named 'fc'
            print(f"Teacher Model Layer {name}:")
            print(param.data.cpu().numpy())

    # Initialize EnMAP encoder and salinity predictor (student model)
    enmap_encoder = SparseStackedAutoencoder(input_dim=enmap_reflectance_shape[1], latent_dim=64)
    enmap_salinity_predictor = TransformerModel(input_dim=64)

    enmap_encoder.to(device)
    enmap_salinity_predictor.to(device)

    optimizer_enmap = optim.Adam(list(enmap_salinity_predictor.parameters()) + list(enmap_encoder.parameters()), lr=lr, weight_decay=weight_decay)
    smooth_l1_loss_criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_enmap, mode='min', factor=0.1, patience=5)

    return pretrained_ftir_salinity_predictor, enmap_encoder, enmap_salinity_predictor, optimizer_enmap, smooth_l1_loss_criterion, scheduler

def test_model(test_loader_paired, enmap_salinity_predictor, pretrained_ftir_salinity_predictor, temperature_scaled_mse_loss, knowledge_distillation_loss, smooth_l1_loss_criterion, device, temperature, alpha):
    """Evaluates the EnMAP salinity predictor model on the test set."""
    enmap_salinity_predictor.eval()
    pretrained_ftir_salinity_predictor.eval()

    test_teacher_loss_epoch = 0
    test_task_loss_epoch = 0
    test_combined_loss_epoch = 0
    test_salinity_mae_epoch = 0
    test_non_zero_mae_epoch = 0
    test_non_zero_loss_epoch = 0

    all_student_preds_test = []
    all_teacher_preds_test = []
    all_enmap_ec_batches_test = []

    with torch.no_grad():
        for batch_idx, (enmap_embeddings_batch_test, enmap_salinity_ec_batch_test) in enumerate(test_loader_paired):
            enmap_embeddings_batch_test = enmap_embeddings_batch_test.to(device)
            enmap_salinity_ec_batch_test = enmap_salinity_ec_batch_test.to(device)

            # Student model predictions
            student_salinity_predictions_test = enmap_salinity_predictor(enmap_embeddings_batch_test)

            # Teacher model predictions
            teacher_salinity_predictions_test = pretrained_ftir_salinity_predictor(enmap_embeddings_batch_test)

            # Compute losses
            test_teacher_loss = temperature_scaled_mse_loss(student_salinity_predictions_test, teacher_salinity_predictions_test, temperature)
            test_task_loss = smooth_l1_loss_criterion(student_salinity_predictions_test, enmap_salinity_ec_batch_test.unsqueeze(1))
            test_combined_loss = knowledge_distillation_loss(student_salinity_predictions_test, teacher_salinity_predictions_test, enmap_salinity_ec_batch_test, temperature, alpha, smooth_l1_loss_criterion)

            # Compute MAE
            test_salinity_mae = mean_absolute_error(
                enmap_salinity_ec_batch_test.cpu().detach().numpy(),
                student_salinity_predictions_test.cpu().detach().numpy()
            )

            # Compute non-zero metrics
            test_non_zero_mae, test_non_zero_loss = compute_non_zero_metrics(student_salinity_predictions_test, enmap_salinity_ec_batch_test)

            # Accumulate losses
            test_teacher_loss_epoch += test_teacher_loss.item()
            test_task_loss_epoch += test_task_loss.item()
            test_combined_loss_epoch += test_combined_loss.item()
            test_salinity_mae_epoch += test_salinity_mae
            test_non_zero_mae_epoch += test_non_zero_mae
            test_non_zero_loss_epoch += test_non_zero_loss

            # Store predictions and actual values for printing in specific epochs
            all_student_preds_test.append(student_salinity_predictions_test.cpu().numpy())
            all_teacher_preds_test.append(teacher_salinity_predictions_test.cpu().numpy())
            all_enmap_ec_batches_test.append(enmap_salinity_ec_batch_test.cpu().numpy())

    num_batches = len(test_loader_paired)
    average_test_teacher_loss = test_teacher_loss_epoch / num_batches
    average_test_task_loss = test_task_loss_epoch / num_batches
    average_test_combined_loss = test_combined_loss_epoch / num_batches
    average_test_salinity_mae = test_salinity_mae_epoch / num_batches
    average_test_non_zero_mae = test_non_zero_mae_epoch / num_batches
    average_test_non_zero_loss = test_non_zero_loss_epoch / num_batches

    return average_test_combined_loss, average_test_teacher_loss, average_test_task_loss, average_test_salinity_mae, average_test_non_zero_mae, average_test_non_zero_loss, all_student_preds_test, all_teacher_preds_test, all_enmap_ec_batches_test

# --- 5. Main Function ---
def main():
    enmap_csv_path = '<DATA_ROOT>/valid_enmap_reflectance_with_ec_mask.csv'
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = 1.0  # Temperature for distillation
    alpha = 0.2  # Weight for distillation loss
    batch_size = 64
    lr = 0.001
    weight_decay = 1e-5
    random_state = 42
    undersample_zeros = True

    print("Starting script with Knowledge Distillation and AOI Split...")
    sys.stdout.flush()

    print(f"CUDA available: {torch.cuda.is_available()}")
    sys.stdout.flush()
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        sys.stdout.flush()

    print("Loading data from CSV...")
    sys.stdout.flush()
    enmap_df = pd.read_csv(enmap_csv_path)

    # Remove outliers using IQR
    print("Removing outliers using IQR...")
    sys.stdout.flush()
    enmap_df = remove_outliers_iqr(enmap_df, 'chorizon.ec_r')

    # Visualize the distribution of salinity values after outlier removal
    plt.hist(enmap_df['chorizon.ec_r'], bins=50)
    plt.title("Distribution of Salinity Values After Outlier Removal")
    plt.xlabel("Salinity (EC)")
    plt.ylabel("Frequency")
    plt.show()

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

    print("Initializing pretrained models and optimizer...")
    sys.stdout.flush()
    pretrained_ftir_salinity_predictor, enmap_encoder, enmap_salinity_predictor, optimizer_enmap, smooth_l1_loss_criterion, scheduler = initialize_models_and_optimizer(enmap_reflectance.shape, device, lr, weight_decay)

    print("Starting training and validation...")
    sys.stdout.flush()
    train_teacher_losses, val_teacher_losses, train_task_losses, val_task_losses, train_combined_losses, val_combined_losses, train_salinity_mae_losses, val_salinity_mae_losses, train_non_zero_mae_losses, val_non_zero_mae_losses, train_non_zero_loss_losses, val_non_zero_loss_losses = train_and_validate(
        num_epochs, train_loader_paired, val_loader_paired, enmap_salinity_predictor, pretrained_ftir_salinity_predictor, optimizer_enmap, temperature_scaled_mse_loss, knowledge_distillation_loss, smooth_l1_loss_criterion, device, temperature, alpha, scheduler
    )

    print("Evaluating on Test Set (AOI split with undersampled zeros)...")
    sys.stdout.flush()
    average_test_combined_loss, average_test_teacher_loss, average_test_task_loss, average_test_salinity_mae, average_test_non_zero_mae, average_test_non_zero_loss, all_student_preds_test, all_teacher_preds_test, all_enmap_ec_batches_test = test_model( # Corrected function call to include non_zero metrics and predictions
        test_loader_paired, enmap_salinity_predictor, pretrained_ftir_salinity_predictor, temperature_scaled_mse_loss, knowledge_distillation_loss, smooth_l1_loss_criterion, device, temperature, alpha
    )

    print(f"Final Test Combined Loss (AOI split, undersampled zeros): {average_test_combined_loss:.4f}, Test Teacher KLD Loss: {average_test_teacher_loss:.4f}, Test Task SmoothL1 Loss: {average_test_task_loss:.4f}, Test Salinity MAE: {average_test_salinity_mae:.4f}, Test Non-Zero MAE: {average_test_non_zero_mae:.4f}, Test Non-Zero Loss: {average_test_non_zero_loss:.4f}") # Corrected print statement to include non_zero metrics
    sys.stdout.flush()

    print(f"--- Test Set Predictions (First Batch from Test Set) ---") # Printing predictions for test set as well, for consistency
    print("Student Predictions (Test Batch 1):", all_student_preds_test[0][:5].flatten())
    print("Teacher Predictions (Test Batch 1):", all_teacher_preds_test[0][:5].flatten())
    print("Actual EnMAP EC (Test Batch 1):", all_enmap_ec_batches_test[0][:5].flatten())
    sys.stdout.flush()


if __name__ == '__main__':
    main()