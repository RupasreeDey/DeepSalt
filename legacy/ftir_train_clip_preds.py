import os
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
import sys

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

# Set random seed and device
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Load pre-trained Sparse Stacked Autoencoder
pretrained_encoder = SparseStackedAutoencoder(input_dim=1765, latent_dim=64).to(device)
pretrained_encoder.load_state_dict(torch.load('ftir_sparse_stacked_autoencoder.pth', map_location=device))
pretrained_encoder.eval()  # Set to evaluation mode

# Load datasets and generate embeddings
def load_datasets():
    csv_folder_path = '<DATA_ROOT>/OPUS/'
    input_data, target_data, meta_data, wavenumb, states = [], [], [], [], []
    target_file_path = '<DATA_ROOT>/input.csv'

    with open(target_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header

        init_depth = 0

        df = pd.read_csv(target_file_path, low_memory=False)  # Ignore DtypeWarning

        for row in csv_reader:
            if row[15] == '':  # Skip rows with missing file paths or target values
                continue

            new_file_path = '<DATA_ROOT>' + row[14]

            if not os.path.exists(new_file_path):
                continue

            try:
                if int(row[10]) >= 2 and int(row[11]) <= 12:
                    init_depth += 1
                    df = pd.read_csv(new_file_path, header=None, names=['WaveNumber', 'SpectralMIR'])
                    spectral_values = df['SpectralMIR'].values.astype(float)
                    if len(spectral_values) != 1765:
                        continue

                    input_data.append(spectral_values)
                    target_value = row[15].strip()
                    target_data.append(float(target_value) if target_value else 0.0)

                    wavenumb.append(np.round(df['WaveNumber'].values.astype(float), 0).astype(int))
                    states.append(row[7])
            except ValueError:
                continue

    input_data = np.array(input_data)
    target_data = np.array(target_data)

    # Split data BEFORE scaling
    train_input, test_input, train_target, test_target = train_test_split(
        input_data, target_data, test_size=0.2, random_state=42)

    # Generate embeddings using pre-trained encoder for both train and test sets
    train_input_tensor = torch.tensor(train_input, dtype=torch.float32).to(device)
    test_input_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, train_embeddings = pretrained_encoder(train_input_tensor)  # Extract latent representation (z)
        _, test_embeddings = pretrained_encoder(test_input_tensor)    # Extract latent representation (z)

        train_embeddings = train_embeddings.cpu().numpy()  # Move to CPU and convert to numpy
        test_embeddings = test_embeddings.cpu().numpy()    # Move to CPU and convert to numpy

    # Print sample original data and embeddings
    print("\nSample Original Data and Embeddings:")
    print("-----------------------------------")
    for i in range(3):  # Print first 3 samples
        print(f"\nSample {i + 1}:")
        print(f"Original Data Shape: {train_input[i].shape}")
        print(f"Original Data: {train_input[i]}")
        print(f"Embedding Shape: {train_embeddings[i].shape}")
        print(f"Embedding: {train_embeddings[i]}")

    # Scale embeddings and target data separately for training and testing sets
    input_scaler = MinMaxScaler()
    train_embeddings_scaled = input_scaler.fit_transform(train_embeddings)  # Fit AND transform train
    test_embeddings_scaled = input_scaler.transform(test_embeddings)        # Transform test using train scaler

    target_scaler = MinMaxScaler()
    train_target_scaled = target_scaler.fit_transform(train_target.reshape(-1, 1)).flatten()  # Fit AND transform train
    test_target_scaled = target_scaler.transform(test_target.reshape(-1, 1)).flatten()        # Transform test using train scaler

    return train_embeddings_scaled, test_embeddings_scaled, train_target_scaled, test_target_scaled, wavenumb, states, input_scaler, target_scaler

# Define TimeSeriesDataset
class TimeSeriesDataset(Dataset):
    def __init__(self, train=True, train_input=None, test_input=None, train_target=None, test_target=None):
        if train:
            self.data = torch.tensor(train_input, dtype=torch.float32).to(device)
            self.targets = torch.tensor(train_target, dtype=torch.float32).to(device)
        else:
            self.data = torch.tensor(test_input, dtype=torch.float32).to(device)
            self.targets = torch.tensor(test_target, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

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
        x = x + self.pe[:x.size(1), :]
        return x

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

# Save and load model functions
def save_model(model, filename='ftir_salinity_model_ssae_clipped.pth'):
    torch.save(model.state_dict(), filename)
    print(f'Model saved to {filename}')
    sys.stdout.flush()

def load_model(model, filename='ftir_salinity_model_ssae_clipped.pth'):
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    print(f'Model loaded from {filename}')
    sys.stdout.flush()
    return model

# Inference function
def inference(model, dataset, sample_index=0, target_scaler=None):
    """
    Perform inference on a specific sample from the test set and inverse transform the prediction.
    """
    # Load the test set data
    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)  # Load all test data
    inputs, targets = next(iter(test_loader))  # Extract inputs and targets
    inputs = inputs.to(device)

    # Select the specific sample
    input_data = inputs[sample_index].unsqueeze(0)  # Add batch dimension for a single sample

    # Perform inference
    model.eval()
    predictions_scaled = model(input_data).squeeze().detach().cpu().numpy()

    # Inverse transform predictions if a target scaler is provided
    if target_scaler is not None:
        predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        original_target = target_scaler.inverse_transform(targets[sample_index].cpu().reshape(-1, 1)).flatten()
    else:
        predictions = predictions_scaled
        original_target = targets[sample_index].item()  # Return scaled target if no scaler

    print(f"Scaled Target Value: {targets[sample_index].item()}")  # Print scaled target
    sys.stdout.flush()
    print(f"Original Target Value: {original_target.item() if target_scaler is not None else original_target}")
    sys.stdout.flush()
    print(f"Scaled Prediction: {predictions_scaled.item()}")
    sys.stdout.flush()

    return predictions, original_target

# Training function
def train_model_with_explainability(model, train_dataset, test_dataset, num_epochs=500, batch_size=64, lr=0.001):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.MSELoss()  # Use MSE loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses, r2_scores = [], [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        for data, targets in train_loader:
            data, targets = data.float().to(device), targets.float().to(device)
            optimizer.zero_grad()
            outputs = model(data).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss, test_r2 = 0, []
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.float().to(device), targets.float().to(device)
                outputs = model(data).squeeze()
                test_loss += criterion(outputs, targets).item()
                test_r2.append(r2_score(targets.cpu(), outputs.cpu()))

        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        r2_scores.append(np.mean(test_r2))
        print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, R²: {r2_scores[-1]:.4f}')
        sys.stdout.flush()

    return train_losses, test_losses, r2_scores

# Main execution
if __name__ == "__main__":
    # Load datasets and scalers
    train_input, test_input, train_target, test_target, wavenumb, states, input_scaler, target_scaler = load_datasets()

    # Create datasets
    train_dataset = TimeSeriesDataset(train=True, train_input=train_input, train_target=train_target)
    test_dataset = TimeSeriesDataset(train=False, test_input=test_input, test_target=test_target)

    # Train the model and incorporate interpretability
    model = TransformerModel(input_dim=64).to(device)  # Updated input_dim to 64
    train_losses, test_losses, r2_scores = train_model_with_explainability(model, train_dataset, test_dataset, num_epochs=500)  # Pass train and test datasets

    # Save the trained model
    save_model(model, 'ftir_salinity_model_ssae_clipped.pth')  # Updated filename

    # Example inference
    # Load the trained model
    model = TransformerModel(input_dim=64).to(device)  # Updated input_dim to 64
    model = load_model(model, 'ftir_salinity_model_ssae_clipped.pth')  # Updated filename

    # Perform inference for the first sample (index 0) and inverse transform
    sample_index = 0
    predictions, original_target = inference(model, test_dataset, sample_index=sample_index, target_scaler=target_scaler)  # Pass target_scaler
    print(f"Predictions for sample {sample_index} (original scale):", predictions)
    sys.stdout.flush()

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.title("Training and Testing Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(r2_scores, label="R² Score")
    plt.legend()
    plt.title("R² Score over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("R²")

    plt.tight_layout()
    plt.savefig("training_results_ftir_salinity_ssae_clipped.png")  # Updated filename
    plt.show()