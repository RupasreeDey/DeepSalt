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
from shapely.geometry import Point
import geopandas as gpd
import rtree
from rtree import index  # For spatial indexing
from sklearn.cluster import KMeans  # For clustering AOIs

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
        z = self.encoder(x)  # Latent representation
        x_recon = self.decoder(z)  # Reconstructed input
        return x_recon, z

# --- 1. Define Models ---
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

class TransformerModelTeacher(nn.Module):
    def __init__(self, input_dim=64, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        super(TransformerModelTeacher, self).__init__()
        # Ensure d_model is divisible by nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Remove the input_projection layer
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, x):
        # Reshape for Transformer
        x = x.unsqueeze(1).permute(1, 0, 2)  # Shape: (sequence_length, batch_size, input_dim)
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

        # Scale output to [0.05, 77] instead of [0.05, 1.1]
        salinity_predictions = 0.05 + (76.95 * salinity_predictions)  # Scale output to [0.05, 77]

        return salinity_predictions, x_mean, intermediate_features  # Return predictions, latent features, and intermediate features

class TransformerModelStudent(nn.Module):
    def __init__(self, input_dim=72, d_model=72, nhead=8, num_encoder_layers=4, dim_feedforward=256, dropout=0.1):
        super(TransformerModelStudent, self).__init__()
        # Ensure d_model is divisible by nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Project input_dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, x):
        # Project input to d_model
        x = self.input_projection(x)  # Shape: (batch_size, sequence_length, d_model)

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

        # Scale output to [0.05, 77] instead of [0.05, 1.1]
        salinity_predictions = 0.05 + (76.95 * salinity_predictions)  # Scale output to [0.05, 77]

        return salinity_predictions, x_mean, intermediate_features  # Return predictions, latent features, and intermediate features

# --- 2. Loss Functions ---
class FeatureDistillationLoss(nn.Module):
    def __init__(self):
        super(FeatureDistillationLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, student_features, teacher_features):
        """
        Compute feature distillation loss between student and teacher intermediate features.
        Args:
            student_features: List of intermediate features from the student model (72-dimensional).
            teacher_features: List of intermediate features from the teacher model (64-dimensional).
        Returns:
            feature_loss: Mean squared error between the first 64 features of the student and the teacher's features.
        """
        total_loss = 0
        for student_feat, teacher_feat in zip(student_features, teacher_features):
            # Extract the first 64 features from the student's intermediate features
            student_feat_64 = student_feat[:, :, :64]  # Shape: (batch_size, sequence_length, 64)
            teacher_feat = teacher_feat[:, :, :64]  # Ensure teacher features are also 64-dimensional

            # Compute MSE loss between the truncated student features and teacher features
            total_loss += self.criterion(student_feat_64, teacher_feat)
        return total_loss / len(student_features)

def knowledge_distillation_loss(student_output, teacher_output, student_features, teacher_features, targets, alpha=0.7, beta=0.3, task_loss_criterion=None, feature_loss_criterion=None):
    """
    Compute combined loss for knowledge distillation.
    Args:
        student_output: Predictions from the student model.
        teacher_output: Predictions from the teacher model.
        student_features: Intermediate features from the student model (72-dimensional).
        teacher_features: Intermediate features from the teacher model (64-dimensional).
        targets: Ground truth targets.
        alpha: Weight for task loss.
        beta: Weight for feature distillation loss.
        task_loss_criterion: Loss function for the task (e.g., Huber loss).
        feature_loss_criterion: Loss function for feature distillation (e.g., MSE).
    Returns:
        combined_loss: Combined loss for knowledge distillation.
        task_loss: Task loss component.
        feature_loss: Feature distillation loss component.
    """
    # Task loss (e.g., Huber loss)
    task_loss = task_loss_criterion(student_output, targets.unsqueeze(1))

    # Feature distillation loss (compare only the first 64 features)
    feature_loss = feature_loss_criterion(student_features, teacher_features)

    # Combine losses
    combined_loss = alpha * task_loss + beta * feature_loss

    return combined_loss, task_loss, feature_loss

# --- 3. Data Loading and Preprocessing ---
def load_data(enmap_csv_path):
    return pd.read_csv(enmap_csv_path)

def preprocess_reflectance_and_locations(enmap_df):
    # Extract reflectance features
    enmap_bands = [f'Mean_Reflectance_Band_{i}' for i in range(1, 225)]
    excluded_bands = [f'Mean_Reflectance_Band_{i}' for i in range(130, 136)]
    enmap_reflectance_cols = [band for band in enmap_bands if band not in excluded_bands]
    enmap_reflectance = enmap_df[enmap_reflectance_cols].values

    # Extract 8 features (excluding 'chorizon.silttotal_r')
    feature_columns = [
        'chorizon.sandtotal_r',
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

def split_data_with_clusters(enmap_df, valid_indices_to_use=None, random_state=42):
    """
    Split data into train (80%), validation (10%), and test (10%) sets using clustering.
    """
    SEED = random_state
    data = enmap_df.iloc[valid_indices_to_use].copy()

    # Create a GeoDataFrame for spatial operations
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))

    # Extract the center coordinates for clustering
    center_coords = gdf[['latitude', 'longitude']].values

    # Perform K-Means clustering to find 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=SEED)
    gdf["cluster"] = kmeans.fit_predict(center_coords)

    # Initialize lists to store the splits
    train_indices = []
    val_indices = []
    test_indices = []

    # Split AOIs in each cluster
    for cluster_id in range(3):
        cluster_df = gdf[gdf["cluster"] == cluster_id]
        train, temp = train_test_split(cluster_df.index, train_size=0.8, random_state=SEED)
        val, test = train_test_split(temp, test_size=0.5, random_state=SEED)
        train_indices.extend(train)
        val_indices.extend(val)
        test_indices.extend(test)

    # Extract the data for each set
    train_data_enmap = gdf.iloc[train_indices]
    valid_data_enmap = gdf.iloc[val_indices]
    test_data_enmap = gdf.iloc[test_indices]

    return train_data_enmap, valid_data_enmap, test_data_enmap

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

def train_epoch(train_loader, student_model, teacher_model, optimizer, task_loss_criterion, feature_loss_criterion, device, alpha=0.7, beta=0.3):
    """
    Train the student model for one epoch with knowledge distillation.
    Args:
        train_loader: DataLoader for training data.
        student_model: The model being trained (student).
        teacher_model: Pretrained model (teacher).
        optimizer: Optimizer for the student model.
        task_loss_criterion: Loss function for the task (e.g., Huber loss).
        feature_loss_criterion: Loss function for feature distillation (e.g., MSE).
        device: Device to use for computation (e.g., 'cuda' or 'cpu').
        alpha: Weight for task loss.
        beta: Weight for feature distillation loss.
    Returns:
        avg_loss: Average loss for the epoch.
    """
    student_model.train()
    teacher_model.eval()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass for student model (full 72-dimensional input)
        student_output, _, student_features = student_model(data)

        # Forward pass for teacher model (only the first 64 dimensions)
        with torch.no_grad():
            teacher_input = data[:, :64]  # Use only the first 64 dimensions (EnMAP embeddings)
            teacher_output, _, teacher_features = teacher_model(teacher_input)

        # Compute combined loss
        combined_loss, task_loss, feature_loss = knowledge_distillation_loss(
            student_output, teacher_output, student_features, teacher_features, target,
            alpha, beta, task_loss_criterion, feature_loss_criterion
        )

        # Backward pass and optimization
        combined_loss.backward()
        optimizer.step()

        total_loss += combined_loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate_epoch(val_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output, _, _ = model(data)
            loss = criterion(output, target.unsqueeze(1))

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def test_epoch(test_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output, _, _ = model(data)
            loss = criterion(output, target.unsqueeze(1))

            total_loss += loss.item()

            # Store predictions and targets for metrics
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    # Calculate metrics
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()

    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))

    return mae, r2, rmse

# --- 4. Main Function ---
def main():
    enmap_csv_path = 'multiplied_enmap_ssurgo_map.csv'
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64  # Reduced batch size for stability
    lr = 0.001  # Reduced learning rate
    weight_decay = 1e-5
    random_state = 42
    undersample_zeros = True

    print("Starting script for EnMAP training with Knowledge Distillation...")
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

    # Split data into train, validation, and test sets using clustering
    print("Splitting data using clustering...")
    sys.stdout.flush()
    train_data_enmap, valid_data_enmap, test_data_enmap = split_data_with_clusters(enmap_df, valid_indices_to_use, random_state)

    # Split data into train, validation, and test sets
    train_reflectance = enmap_reflectance[train_data_enmap.index]
    val_reflectance = enmap_reflectance[valid_data_enmap.index]
    test_reflectance = enmap_reflectance[test_data_enmap.index]

    train_features = features[train_data_enmap.index]
    val_features = features[valid_data_enmap.index]
    test_features = features[test_data_enmap.index]

    train_salinity_ec = enmap_salinity_ec[train_data_enmap.index]
    val_salinity_ec = enmap_salinity_ec[valid_data_enmap.index]
    test_salinity_ec = enmap_salinity_ec[test_data_enmap.index]

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

    # Initialize student model
    print("Initializing student model and optimizer...")
    sys.stdout.flush()
    student_model = TransformerModelStudent(input_dim=train_combined_features.shape[1]).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay)

    # Load pretrained teacher model
    print("Loading pretrained teacher model...")
    sys.stdout.flush()
    teacher_model = TransformerModelTeacher(input_dim=64).to(device)
    teacher_model.load_state_dict(torch.load("ftir_salinity_model_ssae_clipped_final.pth", map_location=device))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Initialize loss functions
    huber_loss_criterion = nn.HuberLoss()  # Task loss
    feature_distillation_loss_criterion = FeatureDistillationLoss()  # Feature distillation loss

    # Training and validation loop with KD
    print("Starting training and validation with Knowledge Distillation...")
    sys.stdout.flush()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    early_stopping_patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train with KD
        train_loss = train_epoch(
            train_loader, student_model, teacher_model, optimizer,
            huber_loss_criterion, feature_distillation_loss_criterion, device,
            alpha=0.7, beta=0.3  # Adjust alpha and beta as needed
        )

        # Validate (no KD)
        val_loss = validate_epoch(val_loader, student_model, huber_loss_criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        sys.stdout.flush()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(student_model.state_dict(), "best_enmap_salinity_predictor_kd.pth")
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
    plt.title("Training and Validation Losses with Knowledge Distillation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("training_results_kd.png")
    plt.show()

    # Test set evaluation
    print("Starting test set evaluation...")
    sys.stdout.flush()
    mae, r2, rmse = test_epoch(test_loader, student_model, huber_loss_criterion, device)

    print(f"Test MAE: {mae:.4f}, Test R^2: {r2:.4f}, Test RMSE: {rmse:.4f}")
    sys.stdout.flush()

    # Save the model
    torch.save(student_model.state_dict(), "enmap_salinity_predictor_kd.pth")
    print("Saved EnMAP salinity predictor model with Knowledge Distillation.")
    sys.stdout.flush()

if __name__ == '__main__':
    main()