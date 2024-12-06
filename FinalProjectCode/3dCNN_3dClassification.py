# ----------------------------
# Environment Setup and Imports
# ----------------------------

import os
import numpy as np
import torch
import lightning.pytorch as pl
import torchmetrics
from torchinfo import summary
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime

# Set the CUDA device
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Get the current time for naming saved files
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Ensure output directories exist
os.makedirs("plots3DCNN", exist_ok=True)

# ----------------------------
# Dataset and DataModule
# ----------------------------

class Numpy3DDataset(Dataset):
    def __init__(self, root_dir, transform=None, dtype=torch.uint8):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.dtype = dtype

        self.classes = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])  # Get class names

        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_path):
                if file_name.endswith(".npy"):  # Only process .npy files
                    file_path = os.path.join(class_path, file_name)
                    self.data.append(file_path)
                    self.labels.append(class_idx)

        if not self.data:
            raise ValueError(f"No '.npy' files found in {root_dir}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        array = np.load(file_path)
        if self.transform:
            array = self.transform(array)
        array = (array > 0).astype(np.uint8)
        array = np.expand_dims(array, axis=0)
        array = torch.from_numpy(array).to(self.dtype)

        # Convert input tensor to float32
        array = array.to(torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return array, label


class Numpy3DDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, val_split=0.1, test_split=0.1, num_workers=0, location="NPYdata", transform=None, dtype=torch.uint8):
        super().__init__()
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.location = location
        self.transform = transform
        self.dtype = dtype

    def setup(self, stage: str = None):
        full_dataset = Numpy3DDataset(root_dir=self.location, transform=self.transform, dtype=self.dtype)
        total_size = len(full_dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - test_size - val_size
        self.data_train, self.data_val, self.data_test = random_split(full_dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

# ----------------------------
#  Model Components (3D CNN)
# ----------------------------

class ThreeDCNN(torch.nn.Module):
    def __init__(self, input_shape, output_size):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(128 * (input_shape[1] // 8) * (input_shape[2] // 8) * (input_shape[3] // 8), 256)
        self.fc2 = torch.nn.Linear(256, output_size)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = x.to(torch.float32)  # Ensure input tensor is float32
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ----------------------------
# Training with Lightning and Confusion Matrix
# ----------------------------

class MultiClassLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes, class_names):
        super().__init__()
        self.model = model
        self.class_names = class_names
        self.network_loss = torch.nn.CrossEntropyLoss()
        self.network_metrics = torch.nn.ModuleDict({
            'acc': torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes),
            'f1': torchmetrics.classification.F1Score(task='multiclass', num_classes=num_classes, average='macro'),
        })
        self.confusion_matrix = torchmetrics.classification.ConfusionMatrix(task='multiclass', num_classes=num_classes)

        # Metrics to track test performance
        self.test_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_f1 = torchmetrics.classification.F1Score(task='multiclass', num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.network_loss(y_pred, y)
        self.log('train_loss', loss)
        acc = self.network_metrics['acc'](y_pred, y)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.network_loss(y_pred, y)
        self.log('val_loss', loss)
        acc = self.network_metrics['acc'](y_pred, y)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.network_loss(y_pred, y)
        acc = self.test_acc(y_pred, y)
        f1 = self.test_f1(y_pred, y)

        # Update confusion matrix
        self.confusion_matrix.update(y_pred, y)

        # Log metrics for test batch
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        # Compute and log confusion matrix
        cm = self.confusion_matrix.compute().cpu().numpy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.savefig(f"Confusion_Matrix_vit_{current_time}.png")
        plt.close()
        print(f"Confusion matrix saved as 'Confusion_Matrix_vit_{current_time}.png'")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# ----------------------------
# Main Training Loop
# ----------------------------

dataset_path = "../NPYdata"
data_module = Numpy3DDataModule(batch_size=8, location=dataset_path, num_workers=0, dtype=torch.uint8)
data_module.setup('fit')

input_shape = data_module.data_train.dataset[0][0].shape
output_size = len(data_module.data_train.dataset.classes)  # Corrected to access classes directly

three_d_cnn = ThreeDCNN(input_shape=input_shape, output_size=output_size)
model = MultiClassLightningModule(three_d_cnn, num_classes=output_size, class_names=data_module.data_train.dataset.classes)

logger = pl.loggers.CSVLogger("logs", name=f"3DCNN_{current_time}")
trainer = pl.Trainer(max_epochs=100, accelerator="gpu", logger=logger)

print("\nStarting Training:")
trainer.fit(model, datamodule=data_module)

print("\nStarting Testing:")
test_results = trainer.test(model, datamodule=data_module)

# Display test metrics
test_acc = test_results[0]['test_acc']
test_loss = test_results[0]['test_loss']
test_f1 = test_results[0]['test_f1']

print(f"\nTest Results:")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

