# pointnet_3d_object_detection.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
from plyfile import PlyData


# PointNet Model Definition
class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.relu = nn.ReLU()

        # Initialize weights to identity matrix
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        identity = torch.eye(k).flatten()
        self.fc3.bias.data.copy_(identity)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(batch_size, self.k, self.k)

        return x


class PointNet(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet, self).__init__()

        self.tnet1 = TNet(k=3)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(2, 1)

        # Apply T-Net for alignment
        trans = self.tnet1(x)
        x = torch.bmm(trans, x)

        # Feature extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Global feature aggregation
        x = torch.max(x, 2)[0]

        # Classification
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Custom Dataset Class

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=1024):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])

        # Load point cloud
        point_cloud = PlyData.read(file_path)['vertex']
        points = np.vstack((point_cloud['x'], point_cloud['y'], point_cloud['z'])).T

        if len(points) > self.num_points:
            points = points[:self.num_points]
        else:
            pad = self.num_points - len(points)
            points = np.pad(points, ((0, pad), (0, 0)), mode='constant')

        label = int(self.files[idx].split('_')[0])  # Assuming filename format: class_index_xxx.ply

        return torch.tensor(points, dtype=torch.float), label


# Chamfer Distance Loss (for reconstruction)

def chamfer_distance(pred, target):
    batch_size, num_points, _ = pred.size()

    dist1 = torch.cdist(pred, target)
    dist2 = torch.cdist(target, pred)

    loss = dist1.min(dim=2)[0].mean() + dist2.min(dim=2)[0].mean()
    return loss


# Train Function

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for points, labels in tqdm(train_loader):
        points, labels = points.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# Evaluate Function

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for points, labels in val_loader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

# =============================
# Main Function
# =============================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="PointNet 3D Object Detection and Reconstruction")
    parser.add_argument('--data', type=str, required=True, help="Path to dataset folder")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    # Load dataset
    train_dataset = PointCloudDataset(args.data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = PointNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, train_loader, device)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        # Save model
        torch.save(model.state_dict(), "best_pointnet.pth")

    print("[INFO] Training complete")

if __name__ == "__main__":
    main()
