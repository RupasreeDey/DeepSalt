import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import matplotlib.pyplot as plt

# Define Sparse Stacked Autoencoder
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

# Sparse Loss (KL Divergence for sparsity)
class SparseLoss(nn.Module):
    def __init__(self, rho=0.05, beta=0.1):
        super(SparseLoss, self).__init__()
        self.rho = rho  # Desired sparsity level
        self.beta = beta  # Weight for sparsity penalty
        self.epsilon = 1e-10  # Small constant to avoid log(0)

    def forward(self, activations):
        # Calculate average activation for each neuron in the hidden layer
        rho_hat = torch.mean(activations, dim=0)
        
        # Clip rho_hat to avoid log(0) or log(1)
        rho_hat = torch.clamp(rho_hat, self.epsilon, 1 - self.epsilon)
        
        # KL Divergence between desired sparsity (rho) and actual sparsity (rho_hat)
        kl_div = self.rho * torch.log(self.rho / rho_hat) + (1 - self.rho) * torch.log((1 - self.rho) / (1 - rho_hat))
        return self.beta * torch.sum(kl_div)

# Combined Loss (Reconstruction + Sparsity)
def combined_loss(recon_x, x, activations, sparse_loss_fn):
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss(reduction='mean')(recon_x, x)
    # Sparsity loss
    sparsity_loss = sparse_loss_fn(activations)
    # Total loss
    return recon_loss + sparsity_loss

# Load data
ftir_data = pd.read_csv('<DATA_ROOT>/ftir_reflectance_with_salinity.csv')
selected_column_names = ftir_data.columns[5:]
print(selected_column_names)
ftir_data_values = ftir_data[selected_column_names].values

# Min-Max Scaling
ftir_scaler = MinMaxScaler()
ftir_data_normalized = ftir_scaler.fit_transform(ftir_data_values)

# Check for NaN, inf, or extreme values in the data
print("Max value in data:", np.max(ftir_data_normalized))
print("Min value in data:", np.min(ftir_data_normalized))
print("NaN values in data:", np.isnan(ftir_data_normalized).sum())
print("Inf values in data:", np.isinf(ftir_data_normalized).sum())

# Convert to PyTorch tensors and dataloaders
ftir_tensor = torch.tensor(ftir_data_normalized, dtype=torch.float32)
ftir_dataset = TensorDataset(ftir_tensor)
ftir_loader = DataLoader(ftir_dataset, batch_size=32, shuffle=True, drop_last=True)

# Initialize Sparse Stacked Autoencoder
input_dim = ftir_tensor.shape[1]  # 1765
latent_dim = 64  # Desired embedding size

sparse_autoencoder = SparseStackedAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
sparse_loss_fn = SparseLoss(rho=0.05, beta=0.1)  # Adjust rho and beta as needed

# Weight Initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

sparse_autoencoder.apply(init_weights)

# Optimizer
optimizer = optim.Adam(sparse_autoencoder.parameters(), lr=0.0001, weight_decay=1e-5)  # Reduced learning rate

# Training loop for Sparse Stacked Autoencoder
num_epochs = 50
train_losses = []

print("Training starting...")
sys.stdout.flush()

for epoch in range(num_epochs):
    sparse_autoencoder.train()
    train_loss = 0
    for (ftir_batch,) in ftir_loader:
        optimizer.zero_grad()
        
        # Forward pass
        ftir_recon, ftir_embedding = sparse_autoencoder(ftir_batch)
        
        # Compute loss
        loss = combined_loss(ftir_recon, ftir_batch, ftir_embedding, sparse_loss_fn)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(sparse_autoencoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
    
    train_losses.append(train_loss / len(ftir_loader))
    print(f'Epoch {epoch+1}, Sparse Stacked AE Loss: {train_losses[-1]:.4f}')
    sys.stdout.flush()

# Save Sparse Stacked Autoencoder model
torch.save(sparse_autoencoder.state_dict(), 'ftir_sparse_stacked_autoencoder.pth')
print("Sparse Stacked Autoencoder trained and saved.")

# Generate FTIR embeddings using the trained Sparse Stacked Autoencoder
sparse_autoencoder.eval()  # Set to evaluation mode
ftir_embeddings = []
with torch.no_grad():  # Disable gradients for embedding generation
    for (ftir_batch,) in ftir_loader:
        _, ftir_embedding = sparse_autoencoder(ftir_batch)
        ftir_embeddings.append(ftir_embedding)

ftir_embeddings = torch.cat(ftir_embeddings, dim=0).numpy()
np.save('ftir_sparse_embeddings.npy', ftir_embeddings)  # Save embeddings
print("FTIR embeddings generated using Sparse Stacked Autoencoder and saved as ftir_sparse_embeddings.npy")

# Function to plot actual vs generated spectra
def plot_actual_vs_generated(actual, generated, sample_indices):
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(sample_indices):
        plt.subplot(5, 1, i+1)
        plt.plot(actual[i], label='Actual Spectrum')  # Use `i` instead of `idx`
        plt.plot(generated[i], label='Generated Spectrum')  # Use `i` instead of `idx`
        plt.title(f'Sample {idx+1}')
        plt.legend()
    plt.tight_layout()
    plt.show()

# Select 5 random test samples
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(len(ftir_tensor), min(5, len(ftir_tensor)), replace=False)
print("Selected sample indices:", sample_indices)

# Generate the reconstructed spectra for the selected samples
sparse_autoencoder.eval()  # Set to evaluation mode
with torch.no_grad():  # Disable gradients for embedding generation
    actual_spectra = ftir_tensor[sample_indices].numpy()
    generated_spectra, _ = sparse_autoencoder(ftir_tensor[sample_indices])
    generated_spectra = generated_spectra.numpy()

# Plot the actual vs generated spectra
plot_actual_vs_generated(actual_spectra, generated_spectra, sample_indices)