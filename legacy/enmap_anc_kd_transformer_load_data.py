import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os
import sys
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
            nn.Linear(256, latent_dim),
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
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# Positional Encoding for Transformer
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

# Teacher Model (Transformer)
class TransformerModelTeacher(nn.Module):
    def __init__(self, input_dim=64, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        super(TransformerModelTeacher, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # Prediction head
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # Input projection and positional encoding
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # Add sequence dimension (seq_len=1)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, features)
        x = self.pos_encoder(x)
        
        # Transformer processing with intermediate features
        intermediate_features = []
        for layer in self.transformer_encoder.layers:
            x = layer(x)
            intermediate_features.append(x)
        
        # Final prediction
        x = x.permute(1, 0, 2)  # Back to (batch, seq_len, features)
        x_mean = x.mean(dim=1)  # Pool across sequence
        salinity_predictions = 0.05 + (76.95 * self.sigmoid(self.fc(x_mean)))
        
        return salinity_predictions, x_mean, intermediate_features

# Student Model (Larger Transformer)
class TransformerModelStudent(nn.Module):
    def __init__(self, input_dim=72, d_model=96, nhead=8, num_encoder_layers=4, dim_feedforward=256, dropout=0.1):
        super(TransformerModelStudent, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        # Enhanced input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder with more layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # Projection layers to match teacher's feature dimensions
        self.feature_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.LayerNorm(64)
            ) for _ in range(num_encoder_layers)
        ])
        
        # Enhanced prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward//2, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # Input processing
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.pos_encoder(x)
        
        # Transformer processing with feature projections
        intermediate_features = []
        for i, layer in enumerate(self.transformer_encoder.layers):
            x = layer(x)
            # Project features to teacher's dimension (64) for distillation
            projected_features = self.feature_projections[i](x)
            intermediate_features.append(projected_features)
        
        # Final prediction
        x = x.permute(1, 0, 2)  # Back to (batch, seq_len, features)
        x_mean = x.mean(dim=1)  # Pool across sequence
        salinity_predictions = 0.05 + (76.95 * self.sigmoid(self.prediction_head(x_mean)))
        
        return salinity_predictions, x_mean, intermediate_features

# Feature Distillation Loss (Modified for Transformer)
class FeatureDistillationLoss(nn.Module):
    def __init__(self, layer_weights=[1.0, 0.8, 0.6], temperature=2.0):
        super().__init__()
        self.layer_weights = layer_weights
        self.temperature = temperature
        self.criterion = nn.MSELoss()
        
    def forward(self, student_features, teacher_features):
        total_loss = 0
        num_layers = min(len(student_features), len(teacher_features))
        
        for layer_idx in range(num_layers):
            weight = self.layer_weights[min(layer_idx, len(self.layer_weights)-1)]
            # Align dimensions: take mean across sequence for student
            s_feat = student_features[layer_idx].mean(dim=0)  # (batch, d_model)
            t_feat = teacher_features[layer_idx].mean(dim=0)  # (batch, d_model)
            loss = self.criterion(s_feat, t_feat)
            total_loss += weight * loss
        
        return total_loss / num_layers

# Knowledge Distillation Loss (same as before)
def knowledge_distillation_loss(student_output, teacher_output, student_features, teacher_features, targets, 
                              alpha=0.35, beta=0.65, task_loss_criterion=None, feature_loss_criterion=None):
    task_loss = task_loss_criterion(student_output, targets.unsqueeze(1))
    feature_loss = feature_loss_criterion(student_features, teacher_features)
    output_distill_loss = F.mse_loss(student_output, teacher_output.detach())
    combined_loss = alpha * task_loss + beta * feature_loss + (1 - alpha - beta) * output_distill_loss
    return combined_loss, task_loss, feature_loss

def generate_embeddings(reflectance, enmap_encoder_pretrained, device):
    reflectance_tensor = torch.tensor(reflectance, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, embeddings = enmap_encoder_pretrained(reflectance_tensor)
        embeddings = embeddings.cpu().numpy()
    return embeddings

def load_preprocessed_data(save_dir='preprocessed_data'):
    """Load all preprocessed data from disk"""
    data_dict = {}
    for file in os.listdir(save_dir):
        if not file.endswith(('.npy', '.pkl')):
            continue
        name = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1]
        if ext == '.npy':
            data_dict[name] = np.load(os.path.join(save_dir, file), allow_pickle=True)
        elif ext == '.pkl':
            if name in ['feature_names', 'reflectance_columns']:
                data_dict[name] = pd.read_pickle(os.path.join(save_dir, file))
            else:
                data_dict[name] = joblib.load(os.path.join(save_dir, file))
    return data_dict

def create_dataloaders_from_saved(batch_size=64, undersample_zeros=True):
    """Create dataloaders from saved preprocessed data"""
    data = load_preprocessed_data()
    
    # Convert to tensors
    X_train = torch.FloatTensor(data['X_train'])
    X_val = torch.FloatTensor(data['X_val'])
    X_test = torch.FloatTensor(data['X_test'])
    y_train = torch.FloatTensor(data['y_train'])
    y_val = torch.FloatTensor(data['y_val'])
    y_test = torch.FloatTensor(data['y_test'])
    
    # Undersample zeros if needed
    if undersample_zeros:
        zero_mask = (y_train == 0)
        nonzero_mask = ~zero_mask
        zero_indices = torch.where(zero_mask)[0]
        keep_zero = int(0.1 * len(zero_indices))
        kept_zero = zero_indices[torch.randperm(len(zero_indices))[:keep_zero]]
        all_indices = torch.cat([kept_zero, torch.where(nonzero_mask)[0]])
        X_train = X_train[all_indices]
        y_train = y_train[all_indices]
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, data

def train_epoch(train_loader, student_model, teacher_model, optimizer, task_loss_criterion, feature_loss_criterion, device, alpha=0.1, beta=0.9):
    student_model.train()
    teacher_model.eval()
    total_combined_loss = 0.0
    total_task_loss = 0.0
    total_feature_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        student_output, _, student_features = student_model(data)
        with torch.no_grad():
            teacher_input = data[:, :64]
            teacher_output, _, teacher_features = teacher_model(teacher_input)

        combined_loss, task_loss, feature_loss = knowledge_distillation_loss(
            student_output, teacher_output, student_features, teacher_features, target,
            alpha, beta, task_loss_criterion, feature_loss_criterion
        )

        combined_loss.backward()
        optimizer.step()

        total_combined_loss += combined_loss.item()
        total_task_loss += task_loss.item()
        total_feature_loss += feature_loss.item()

    return total_combined_loss / len(train_loader), total_task_loss / len(train_loader), total_feature_loss / len(train_loader)

def validate_epoch(val_loader, student_model, teacher_model, task_loss_criterion, feature_loss_criterion, device, alpha=0.1, beta=0.9):
    student_model.eval()
    teacher_model.eval()
    total_combined_loss = 0.0
    total_task_loss = 0.0
    total_feature_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            student_output, _, student_features = student_model(data)
            teacher_input = data[:, :64]
            teacher_output, _, teacher_features = teacher_model(teacher_input)

            combined_loss, task_loss, feature_loss = knowledge_distillation_loss(
                student_output, teacher_output, student_features, teacher_features, target,
                alpha, beta, task_loss_criterion, feature_loss_criterion
            )

            total_combined_loss += combined_loss.item()
            total_task_loss += task_loss.item()
            total_feature_loss += feature_loss.item()

    return total_combined_loss / len(val_loader), total_task_loss / len(val_loader), total_feature_loss / len(val_loader)

def test_epoch(test_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            loss = criterion(output, target.unsqueeze(1))
            total_loss += loss.item()
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    return mae, r2, rmse

def main():
    # Configuration
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    lr = 0.001
    weight_decay = 1e-5
    layer_weights = [1.35, 1.05, 0.65]
    alpha = 0.35
    beta = 0.65

    print("Starting training with knowledge distillation...")
    print(f"Using device: {device}")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_loader, val_loader, test_loader, data = create_dataloaders_from_saved(batch_size=batch_size)
    
    # Initialize models - CHANGED TO TRANSFORMERS
    print("Initializing transformer models...")
    student_model = TransformerModelStudent(input_dim=data['X_train'].shape[1]).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay)

    teacher_model = TransformerModelTeacher(input_dim=64).to(device)
    teacher_state_dict = torch.load("ftir_salinity_model_ssae_clipped_final.pth", map_location=device)
    teacher_model.load_state_dict(teacher_state_dict)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Loss functions
    huber_loss_criterion = nn.HuberLoss()
    feature_distillation_loss_criterion = FeatureDistillationLoss(layer_weights=layer_weights)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training loop
    print("Starting training...")
    sys.stdout.flush()

    train_losses, val_losses = [], []
    train_task_losses, val_task_losses = [], []
    train_feature_losses, val_feature_losses = [], []
    best_val_loss = float('inf')
    early_stopping_patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        train_combined_loss, train_task_loss, train_feature_loss = train_epoch(
            train_loader, student_model, teacher_model, optimizer,
            huber_loss_criterion, feature_distillation_loss_criterion, device,
            alpha, beta
        )

        # Validate
        val_combined_loss, val_task_loss, val_feature_loss = validate_epoch(
            val_loader, student_model, teacher_model,
            huber_loss_criterion, feature_distillation_loss_criterion, device,
            alpha, beta
        )

        # Store metrics
        train_losses.append(train_combined_loss)
        val_losses.append(val_combined_loss)
        train_task_losses.append(train_task_loss)
        val_task_losses.append(val_task_loss)
        train_feature_losses.append(train_feature_loss)
        val_feature_losses.append(val_feature_loss)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_combined_loss:.4f} "
              f"(Task: {train_task_loss:.4f}, "
              f"Feature: {train_feature_loss:.4f}) | "
              f"Val Loss: {val_combined_loss:.4f} "
              f"(Task: {val_task_loss:.4f}, "
              f"Feature: {val_feature_loss:.4f})")
        
        sys.stdout.flush()
        
        # Early stopping
        scheduler.step(val_combined_loss)
        if val_combined_loss < best_val_loss:
            best_val_loss = val_combined_loss
            patience_counter = 0
            torch.save(student_model.state_dict(), "best_student_model_resnet_kd.pth")
            print("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Plotting results
    print("Plotting training results...")
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.title("Combined Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(train_task_losses, label="Train")
    plt.plot(val_task_losses, label="Val")
    plt.legend()
    plt.title("Task Loss")
    
    plt.subplot(1, 3, 3)
    plt.plot(train_feature_losses, label="Train")
    plt.plot(val_feature_losses, label="Val")
    plt.legend()
    plt.title("Feature Loss")
    
    plt.tight_layout()
    plt.savefig("training_results_transformer_kd.png")
    plt.show()

    # Test evaluation
    print("Evaluating on test set...")
    mae, r2, rmse = test_epoch(test_loader, student_model, huber_loss_criterion, device)
    print(f"Test Metrics - MAE: {mae:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}")
    sys.stdout.flush()

    # Save final model
    torch.save(student_model.state_dict(), "final_student_model_resnet_kd.pth")
    print("Training complete. Models saved.")

if __name__ == '__main__':
    main()
