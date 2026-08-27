import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
import sys
import random

# Set random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# --- 1. Define Sparse Stacked Autoencoder (SSAE) for EnMAP ---
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

# --- 2. Combined Loss (Reconstruction + Cosine Similarity) ---
def combined_loss(recon_x, x, enmap_embeddings, ftir_embeddings, recon_loss_factor, contrastive_loss_factor):
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss(reduction='mean')(recon_x, x)
    # Normalize embeddings
    enmap_embeddings_norm = nn.functional.normalize(enmap_embeddings, p=2, dim=1)
    ftir_embeddings_norm = nn.functional.normalize(ftir_embeddings, p=2, dim=1)
    # Cosine similarity loss (1 - cosine similarity)
    cosine_sim = torch.sum(enmap_embeddings_norm * ftir_embeddings_norm, dim=1)
    cosine_loss = 1 - cosine_sim.mean()
    # Total loss
    return recon_loss_factor * recon_loss + contrastive_loss_factor * cosine_loss, recon_loss, cosine_loss

# --- 3. Data Loading and Preprocessing Functions ---
def load_data(ftir_csv_path, enmap_csv_path):
    ftir_df = pd.read_csv(ftir_csv_path)
    enmap_df = pd.read_csv(enmap_csv_path)
    return ftir_df, enmap_df

def preprocess_data(ftir_df, enmap_df):
    # Select EnMAP bands: Mean_Band_1 to Mean_Band_224, excluding Mean_Band_130 to Mean_Band_135
    enmap_bands = [f'Mean_Reflectance_Band_{i}' for i in range(1, 225)]
    excluded_bands = [f'Mean_Reflectance_Band_{i}' for i in range(130, 136)]
    selected_enmap_bands = [band for band in enmap_bands if band not in excluded_bands]
    enmap_reflectance = enmap_df[selected_enmap_bands].values

    # Select FTIR reflectance columns (assuming columns from index 5 onwards)
    ftir_reflectance_cols = ftir_df.columns[5:].tolist()
    ftir_reflectance = ftir_df[ftir_reflectance_cols].values

    # Extract locations
    ftir_locations = ftir_df[['latitude', 'longitude']].values
    enmap_locations = enmap_df[['latitude', 'longitude']].values

    # Check for NaN values in locations and remove corresponding rows
    ftir_valid_indices = ~np.isnan(ftir_locations).any(axis=1)
    enmap_valid_indices = ~np.isnan(enmap_locations).any(axis=1)

    ftir_reflectance = ftir_reflectance[ftir_valid_indices]
    ftir_locations = ftir_locations[ftir_valid_indices]
    enmap_reflectance = enmap_reflectance[enmap_valid_indices]
    enmap_locations = enmap_locations[enmap_valid_indices]

    # Min-Max Scaling
    enmap_scaler = MinMaxScaler()
    enmap_reflectance = enmap_scaler.fit_transform(enmap_reflectance)
    ftir_scaler = MinMaxScaler()
    ftir_reflectance = ftir_scaler.fit_transform(ftir_reflectance)

    return ftir_reflectance, enmap_reflectance, ftir_locations, enmap_locations

def create_paired_data(ftir_reflectance, enmap_reflectance, ftir_locations, enmap_locations, max_distance=0.1):
    """
    Pair FTIR and EnMAP samples based on nearest geographical locations.
    """
    # Use BallTree for efficient nearest-neighbor search
    tree = BallTree(enmap_locations, metric='haversine')  # Use haversine for geographical distances
    paired_indices = []
    paired_enmap_reflectance = []
    paired_ftir_reflectance = []
    used_enmap_indices = set()

    for ftir_idx, ftir_loc in enumerate(ftir_locations):
        # Find the nearest EnMAP sample
        dist, ind = tree.query([ftir_loc], k=1)  # k=1 for the nearest neighbor
        nearest_enmap_idx = ind[0][0]
        nearest_distance = dist[0][0]

        # Check if the nearest EnMAP sample is within the maximum allowed distance
        if nearest_distance <= max_distance and nearest_enmap_idx not in used_enmap_indices:
            paired_indices.append((ftir_idx, nearest_enmap_idx))
            paired_ftir_reflectance.append(ftir_reflectance[ftir_idx])
            paired_enmap_reflectance.append(enmap_reflectance[nearest_enmap_idx])
            used_enmap_indices.add(nearest_enmap_idx)

    # Convert lists to numpy arrays
    paired_enmap_reflectance_np = np.array(paired_enmap_reflectance)
    paired_ftir_reflectance_np = np.array(paired_ftir_reflectance)
    return paired_enmap_reflectance_np, paired_ftir_reflectance_np, paired_indices

def split_data_indices(paired_indices):
    """
    Split paired indices into training, validation, and test sets.
    """
    train_val_indices, test_indices = train_test_split(np.arange(len(paired_indices)), test_size=0.15, random_state=random_seed)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.15/0.85, random_state=random_seed)
    return train_indices, val_indices, test_indices

def create_dataloaders(paired_enmap_reflectance, paired_ftir_reflectance, train_indices, val_indices, test_indices, batch_size=32):
    """
    Create PyTorch DataLoaders for training, validation, and test sets.
    """
    # Training Data
    train_enmap_reflectance_tensor = torch.tensor(paired_enmap_reflectance[train_indices], dtype=torch.float32)
    train_ftir_reflectance_tensor = torch.tensor(paired_ftir_reflectance[train_indices], dtype=torch.float32)
    train_dataset = TensorDataset(train_enmap_reflectance_tensor, train_ftir_reflectance_tensor)
    train_loader_paired = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Validation Data
    val_enmap_reflectance_tensor = torch.tensor(paired_enmap_reflectance[val_indices], dtype=torch.float32)
    val_ftir_reflectance_tensor = torch.tensor(paired_ftir_reflectance[val_indices], dtype=torch.float32)
    val_dataset = TensorDataset(val_enmap_reflectance_tensor, val_ftir_reflectance_tensor)
    val_loader_paired = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Test Data
    test_enmap_reflectance_tensor = torch.tensor(paired_enmap_reflectance[test_indices], dtype=torch.float32)
    test_ftir_reflectance_tensor = torch.tensor(paired_ftir_reflectance[test_indices], dtype=torch.float32)
    test_dataset = TensorDataset(test_enmap_reflectance_tensor, test_ftir_reflectance_tensor)
    test_loader_paired = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader_paired, val_loader_paired, test_loader_paired

def visualize_pairings(ftir_locations, enmap_locations, paired_indices):
    """
    Visualize the geographical locations of paired FTIR and EnMAP samples.
    """
    plt.figure(figsize=(10, 6))
    # Plot FTIR and EnMAP locations
    plt.scatter(ftir_locations[:, 1], ftir_locations[:, 0], c='blue', label='FTIR')
    plt.scatter(enmap_locations[:, 1], enmap_locations[:, 0], c='red', label='EnMAP')

    # Highlight paired samples
    for ftir_idx, enmap_idx in paired_indices:
        plt.plot([ftir_locations[ftir_idx, 1], enmap_locations[enmap_idx, 1]],
                 [ftir_locations[ftir_idx, 0], enmap_locations[enmap_idx, 0]],
                 'k--', alpha=0.5)

    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Paired FTIR and EnMAP Samples')
    plt.savefig('pairs.png')

# --- 4. Training Loop ---
def train_epoch(epoch, num_epochs, train_loader_paired, enmap_encoder, ftir_encoder, optimizer, recon_loss_factor, contrastive_loss_factor, device):
    enmap_encoder.train()
    ftir_encoder.train()
    train_loss = 0
    train_recon_loss = 0
    train_contrastive_loss = 0

    for batch_idx, (enmap_batch, ftir_batch) in enumerate(train_loader_paired):
        enmap_batch = enmap_batch.to(device)
        ftir_batch = ftir_batch.to(device)

        optimizer.zero_grad()

        # Forward pass for EnMAP
        enmap_recon, enmap_embeddings = enmap_encoder(enmap_batch)
        # Forward pass for FTIR
        ftir_recon, ftir_embeddings = ftir_encoder(ftir_batch)

        # Compute loss
        loss, recon_loss, contrastive_loss = combined_loss(enmap_recon, enmap_batch, enmap_embeddings, ftir_embeddings, recon_loss_factor, contrastive_loss_factor)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_contrastive_loss += contrastive_loss.item()

    avg_train_loss = train_loss / len(train_loader_paired)
    avg_train_recon_loss = train_recon_loss / len(train_loader_paired)
    avg_train_contrastive_loss = train_contrastive_loss / len(train_loader_paired)
    return avg_train_loss, avg_train_recon_loss, avg_train_contrastive_loss

# --- 5. Main Function ---
def main(num_epochs):
    # Load data
    ftir_df, enmap_df = load_data('ftir_reflectance_with_salinity.csv', 'multiplied_enmap_ssurgo_map.csv')
    ftir_reflectance, enmap_reflectance, ftir_locations, enmap_locations = preprocess_data(ftir_df, enmap_df)

    # Create paired data
    paired_enmap_reflectance, paired_ftir_reflectance, paired_indices = create_paired_data(
        ftir_reflectance, enmap_reflectance, ftir_locations, enmap_locations, max_distance=0.01
    )

    # Visualize pairings
    visualize_pairings(ftir_locations, enmap_locations, paired_indices)

    # Split data
    train_indices, val_indices, test_indices = split_data_indices(paired_indices)
    train_loader_paired, val_loader_paired, test_loader_paired = create_dataloaders(paired_enmap_reflectance, paired_ftir_reflectance, train_indices, val_indices, test_indices)

    # Initialize models
    input_dim_enmap = paired_enmap_reflectance.shape[1]
    input_dim_ftir = paired_ftir_reflectance.shape[1]
    latent_dim = 64

    enmap_encoder = SparseStackedAutoencoder(input_dim=input_dim_enmap, latent_dim=latent_dim)
    ftir_encoder = SparseStackedAutoencoder(input_dim=input_dim_ftir, latent_dim=latent_dim)

    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enmap_encoder.to(device)
    ftir_encoder.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(list(enmap_encoder.parameters()) + list(ftir_encoder.parameters()), lr=0.0001, weight_decay=1e-5)

    # Training loop
    recon_loss_factor = 1.0
    contrastive_loss_factor = 48.0  # Adjust as needed

    for epoch in range(num_epochs):
        avg_train_loss, avg_train_recon_loss, avg_train_contrastive_loss = train_epoch(
            epoch, num_epochs, train_loader_paired, enmap_encoder, ftir_encoder, optimizer, recon_loss_factor, contrastive_loss_factor, device
        )
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Recon Loss: {avg_train_recon_loss:.4f}, Contrastive Loss: {avg_train_contrastive_loss:.4f}')
        sys.stdout.flush()

    # Save models
    torch.save(enmap_encoder.state_dict(), 'enmap_encoder_ssae_cosine.pth')
    torch.save(ftir_encoder.state_dict(), 'ftir_encoder_ssae_cosine.pth')
    print("Models saved.")

if __name__ == "__main__":
    main(num_epochs=50)