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

# Define Residual Block for ResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_features, out_features, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

# Teacher Model
class ResNetModel(nn.Module):
    def __init__(self, input_dim=64, num_blocks=[2, 2, 2], num_classes=1):
        super(ResNetModel, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
        
        self.adapt1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.adapt2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.adapt3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        features = []
        x = self.layer1(x)
        features.append(self.adapt1(x))
        x = self.layer2(x)
        features.append(self.adapt2(x))
        x = self.layer3(x)
        features.append(self.adapt3(x))
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = 0.05 + (76.95 * x)
        return x, None, features

# Student Model
class ResNetModelStudent(nn.Module):
    def __init__(self, input_dim=72, num_blocks=[2, 2, 2], num_classes=1):
        super(ResNetModelStudent, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
        
        self.adapt1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.adapt2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.adapt3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        features = []
        x = self.layer1(x)
        features.append(self.adapt1(x))
        x = self.layer2(x)
        features.append(self.adapt2(x))
        x = self.layer3(x)
        features.append(self.adapt3(x))
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = 0.05 + (76.95 * x)
        return x, None, features

# Feature Distillation Loss
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
            s_att = F.adaptive_avg_pool1d(student_features[layer_idx], 1).squeeze(-1)
            t_att = F.adaptive_avg_pool1d(teacher_features[layer_idx], 1).squeeze(-1)
            loss = self.criterion(s_att, t_att)
            total_loss += weight * loss
        
        return total_loss / num_layers

# Knowledge Distillation Loss
def knowledge_distillation_loss(student_output, teacher_output, student_features, teacher_features, targets, 
                              alpha=0.1, beta=0.9, task_loss_criterion=None, feature_loss_criterion=None):
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
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    lr = 0.001
    weight_decay = 1e-5
    layer_weights = [1.35, 1.05, 0.65]
    alpha = 0.50
    beta = 0.50

    print("Starting training with knowledge distillation...")
    print(f"Using device: {device}")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_loader, val_loader, test_loader, data = create_dataloaders_from_saved(batch_size=batch_size)
    
    # Initialize models
    print("Initializing models...")
    student_model = ResNetModelStudent(input_dim=data['X_train'].shape[1]).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay)

    teacher_model = ResNetModel(input_dim=64).to(device)
    teacher_state_dict = torch.load("ftir_salinity_model_ssae_clipped_resnet.pth", map_location=device)
    teacher_model.load_state_dict(teacher_state_dict, strict=False)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Loss functions
    huber_loss_criterion = nn.HuberLoss()
    feature_distillation_loss_criterion = FeatureDistillationLoss(layer_weights=layer_weights)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training loop
    print("Starting training...")
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
    plt.savefig("training_results_resnet_kd.png")
    plt.show()

    # Test evaluation
    print("Evaluating on test set...")
    mae, r2, rmse = test_epoch(test_loader, student_model, huber_loss_criterion, device)
    print(f"Test Metrics - MAE: {mae:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}")

    # Save final model
    torch.save(student_model.state_dict(), "final_student_model_resnet_kd.pth")
    print("Training complete. Models saved.")

if __name__ == '__main__':
    main()
