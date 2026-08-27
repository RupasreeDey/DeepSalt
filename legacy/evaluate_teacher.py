import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import sys


# --- 1. Define Teacher Model ---
class TransformerModel(nn.Module):
    def __init__(self, input_dim=64, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)  # PositionalEncoding module
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(d_model, 1)  # Outputs a single salinity value
        self.activation = nn.ReLU()  # Ensure non-negative output

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.pos_encoder(x)  # Apply positional encoding
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2).mean(dim=1)  # Apply transformer and average over sequence
        x = self.fc(x)
        x = self.activation(x)  # Apply ReLU to ensure non-negative output
        return x

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

# --- 2. Load Pre-trained Teacher Model ---
def load_teacher_model(device):
    """Load the pre-trained teacher model."""
    teacher_model = TransformerModel(input_dim=64)
    checkpoint = torch.load('ftir_salinity_model_ssae.pth', map_location=device)
    teacher_model.load_state_dict(checkpoint)
    teacher_model.eval()
    teacher_model.to(device)
    return teacher_model

# --- 3. Data Loading and Preprocessing ---
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

def generate_embeddings(enmap_reflectance, enmap_encoder_pretrained, device):
    """Generates embeddings for EnMAP reflectance data using pretrained encoders."""
    enmap_reflectance_tensor = torch.tensor(enmap_reflectance, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, enmap_embeddings = enmap_encoder_pretrained(enmap_reflectance_tensor)  # Extract latent representation (z)
        enmap_embeddings = enmap_embeddings.cpu().numpy()  # Move to CPU and convert to numpy
    return enmap_embeddings

# --- 4. Evaluate Teacher Model ---
def evaluate_teacher_model(teacher_model, enmap_embeddings, enmap_salinity_ec, device):
    """Evaluate the teacher model on EnMAP embeddings."""
    teacher_model.eval()
    enmap_embeddings_tensor = torch.tensor(enmap_embeddings, dtype=torch.float32).to(device)
    enmap_salinity_ec_tensor = torch.tensor(enmap_salinity_ec, dtype=torch.float32).to(device)

    with torch.no_grad():
        teacher_predictions = teacher_model(enmap_embeddings_tensor)

    # Compute MAE
    mae = mean_absolute_error(enmap_salinity_ec_tensor.cpu().numpy(), teacher_predictions.cpu().numpy())
    print(f"Teacher Model MAE on EnMAP Data: {mae:.4f}")
    sys.stdout.flush()

    # Plot predictions vs ground truth
    plt.scatter(enmap_salinity_ec_tensor.cpu().numpy(), teacher_predictions.cpu().numpy(), alpha=0.5)
    plt.xlabel("Ground Truth Salinity (EC)")
    plt.ylabel("Teacher Model Predictions")
    plt.title("Teacher Model Predictions vs Ground Truth")
    plt.show()

# --- 5. Main Function ---
def main():
    enmap_csv_path = '<DATA_ROOT>/valid_enmap_reflectance_with_ec_mask.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data from CSV...")
    enmap_df = pd.read_csv(enmap_csv_path)

    # Preprocess EnMAP data
    enmap_reflectance, enmap_locations, enmap_salinity_ec, enmap_aoi, valid_indices_to_use = preprocess_reflectance_and_locations(enmap_df)

    # Load pre-trained EnMAP encoder
    enmap_encoder_pretrained = SparseStackedAutoencoder(input_dim=enmap_reflectance.shape[1], latent_dim=64)
    enmap_encoder_pretrained.load_state_dict(torch.load('enmap_ssae.pth', map_location=device))
    enmap_encoder_pretrained.eval()
    enmap_encoder_pretrained.to(device)

    # Generate embeddings
    enmap_embeddings = generate_embeddings(enmap_reflectance, enmap_encoder_pretrained, device)

    # Load teacher model
    teacher_model = load_teacher_model(device)

    # Evaluate teacher model
    evaluate_teacher_model(teacher_model, enmap_embeddings, enmap_salinity_ec, device)

if __name__ == '__main__':
    main()