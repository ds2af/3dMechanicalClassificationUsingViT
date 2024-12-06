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
from PIL import Image
# Set the CUDA device
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datetime import datetime

# Get the current time
current_time = datetime.now()

# ----------------------------
# ----------------------------
# Dataset and DataModule
# ----------------------------

class Numpy3DDataset(Dataset):
    def __init__(self, root_dir, transform=None, dtype=torch.uint8):
        """
        Custom Dataset for loading 3D numpy arrays with data type conversion.

        Args:
            root_dir (str): Path to the root directory containing subfolders for each class.
            transform (callable, optional): Optional transform to be applied on a sample.
            dtype (torch.dtype, optional): Desired data type for the tensor. Defaults to torch.uint8.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.dtype = dtype

        # Only include directories as classes
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
        try:
            array = np.load(file_path)  # Load numpy array
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")

        if self.transform:
            array = self.transform(array)

        # Ensure data is binary
        array = (array > 0).astype(np.uint8)  # Convert to binary if not already

        # Add channel dimension
        array = np.expand_dims(array, axis=0)  # Shape: [1, D, H, W]

        # Convert to PyTorch tensor with specified dtype
        array = torch.from_numpy(array).to(self.dtype)
        label = torch.tensor(label, dtype=torch.long)

        return array, label

class Numpy3DDataModule(pl.LightningDataModule):
    def __init__(self, 
                 batch_size=8, 
                 val_split=0.1,  # 10% for validation
                 test_split=0.1,  # 10% for testing
                 num_workers=0,  # Set to 0 to avoid shared memory issues
                 location="NPYdata", 
                 transform=None,
                 dtype=torch.uint8):
        """
        PyTorch Lightning DataModule for managing 3D numpy datasets with data type specification.

        Args:
            batch_size (int): Number of samples per batch.
            val_split (float): Fraction of data to use for validation.
            test_split (float): Fraction of data to use for testing.
            num_workers (int): Number of subprocesses to use for data loading.
            location (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            dtype (torch.dtype, optional): Desired data type for the tensors. Defaults to torch.uint8.
        """
        super().__init__()
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.location = location
        self.transform = transform
        self.dtype = dtype
        self.input_shape = None
        self.output_shape = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.classes = None

    def setup(self, stage: str = None):
        # Load the full dataset
        full_dataset = Numpy3DDataset(root_dir=self.location, transform=self.transform, dtype=self.dtype)
        
        # Print dataset size
        print(f"Total samples: {len(full_dataset)}")

        # Set input/output shapes
        sample_array, _ = full_dataset[0]  # Get a single sample
        self.input_shape = sample_array.shape  # Should be [1, D, H, W]
        self.output_shape = len(full_dataset.classes)  # Number of classes
        self.classes = full_dataset.classes  # Store class names

        print(f"Input shape: {self.input_shape}")
        print(f"Number of classes: {self.output_shape}")

        # Calculate lengths for splits
        total_size = len(full_dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - test_size - val_size

        if stage in ['fit', None]:  # Handle stage=None for some Lightning versions
            self.data_train, self.data_val, self.data_test = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                # generator=torch.Generator().manual_seed(42)  # For reproducibility
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False  # Speeds up data transfer to GPU
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test, 
            batch_size=1,  # For single sample prediction
            num_workers=self.num_workers, 
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

# ----------------------------
# Model Components
# ----------------------------

class SinePositionEmbedding(torch.nn.Module):
    def __init__(self, hidden_size, max_len=512):
        """
        Generates sine-cosine positional embeddings for transformer inputs.

        Args:
            hidden_size (int): Size of the hidden layer (latent_size).
            max_len (int): Maximum sequence length (number of patches).
        """
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, hidden_size]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional embeddings to the input tensor.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            Tensor: Positionally encoded tensor.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.pe.size(1)}")
        return self.pe[:, :seq_len, :]

class TransformerBlock(torch.nn.Module):
    def __init__(self, latent_size=256, num_heads=4, dropout=0.1):
        """
        Transformer Block comprising Multi-Head Attention and Feed-Forward layers.

        Args:
            latent_size (int): Size of the latent vector.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.layer_norm1 = torch.nn.LayerNorm(latent_size)
        self.layer_norm2 = torch.nn.LayerNorm(latent_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()
        self.linear = torch.nn.Linear(latent_size, latent_size)
        self.mha = torch.nn.MultiheadAttention(latent_size, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        """
        Forward pass of the Transformer Block.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, latent_size].

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, latent_size].
        """
        y = self.layer_norm1(x)
        y = self.mha(y, y, y)[0]
        x = x + y
        y = self.layer_norm2(x)
        y = self.linear(y)
        y = self.dropout(y)
        y = self.activation(y)
        return x + y

# Define Trainable Module (Abstract Base Class)
class LightningBoilerplate(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Abstract base class for Lightning modules.
        """
        super().__init__(**kwargs)  # Call the super class constructor

    def predict_step(self, predict_batch, batch_idx):
        x, _ = predict_batch
        y_pred = self.predict(x)
        return y_pred

    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        y_pred = self(x)
        loss = self.network_loss(y_pred, y_true)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        # Compute and log accuracy
        acc = self.network_metrics['acc'](y_pred, y_true)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        y_pred = self(x)
        loss = self.network_loss(y_pred, y_true)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        # Compute and log accuracy
        acc = self.network_metrics['acc'](y_pred, y_true)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y_true = test_batch
        y_pred = self(x)
        loss = self.network_loss(y_pred, y_true)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        # Compute and log accuracy and F1 score
        acc = self.network_metrics['acc'](y_pred, y_true)
        f1 = self.network_metrics['f1'](y_pred, y_true)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True)
        # Convert logits to class indices
        preds = torch.argmax(y_pred, dim=1)
        # Update confusion matrix
        self.confusion_matrix(preds, y_true)
        return loss

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch to compute and plot the confusion matrix.
        """
        # Compute confusion matrix
        cm = self.confusion_matrix.compute().cpu().numpy()
        self.confusion_matrix.reset()

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.trainer.datamodule.classes,
                    yticklabels=self.trainer.datamodule.classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig(f"CM_curve_{current_time}.png")  # Save the figure
        plt.close()

        print(f"Confusion matrix saved as CM_curve_cnn_{current_time}.png")

# Attach loss, metrics, and optimizer
class MultiClassLightningModule(LightningBoilerplate):
    def __init__(self, num_classes, **kwargs):
        """
        Multi-class classification module.

        Args:
            num_classes (int): Number of classes for classification.
        """
        super().__init__(**kwargs)  # Call the super class constructor

        # Initialize accuracy and F1 score metrics
        self.network_metrics = torch.nn.ModuleDict({
            'acc': torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes),
            'f1': torchmetrics.classification.F1Score(task='multiclass', num_classes=num_classes, average='macro')
        })
        
        # Initialize loss function
        self.network_loss = torch.nn.CrossEntropyLoss()
        
        # Initialize confusion matrix with 'multiclass' task
        self.confusion_matrix = torchmetrics.classification.ConfusionMatrix(task='multiclass', num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Attach data type conversion without normalization
class StandardizeTransformModule(MultiClassLightningModule):
    def __init__(self, num_classes, **kwargs):
        """
        Module for converting data types without normalization.

        Args:
            num_classes (int): Number of classes for classification.
        """
        super().__init__(num_classes=num_classes, **kwargs)

    def forward(self, x):
        """
        Converts input tensor from torch.uint8 to torch.float16 for processing.

        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, depth, height, width].

        Returns:
            Tensor: Converted tensor of shape [batch_size, channels, depth, height, width].
        """
        # Convert to float16 for processing
        x = x.to(torch.float16)
        return x

class ViTNetwork(StandardizeTransformModule):
    def __init__(self,
                 input_shape,
                 patch_shape,
                 output_size,
                 latent_size=256,
                 num_heads=4,
                 n_layers=6):
        """
        Vision Transformer for 3D data.

        Args:
            input_shape (tuple): Shape of the input data (channels, depth, height, width).
            patch_shape (tuple): Shape of each patch (depth, height, width).
            output_size (int): Number of output classes.
            latent_size (int): Size of the latent vector.
            num_heads (int): Number of attention heads.
            n_layers (int): Number of transformer layers.
        """
        super().__init__(num_classes=output_size)
        self.save_hyperparameters()

        # Patching: Conv3d with kernel_size=patch_shape and stride=patch_shape
        self.patches = torch.nn.Conv3d(in_channels=input_shape[0],  # channels, likely 1
                                      out_channels=latent_size,
                                      kernel_size=patch_shape,
                                      stride=patch_shape,
                                      bias=False)
        # Compute the number of patches
        D_patches = input_shape[1] // patch_shape[0]
        H_patches = input_shape[2] // patch_shape[1]
        W_patches = input_shape[3] // patch_shape[2]
        num_patches = D_patches * H_patches * W_patches

        self.position_embedding = SinePositionEmbedding(hidden_size=latent_size, max_len=num_patches)
        self.transformer_blocks = torch.nn.Sequential(*[
            TransformerBlock(latent_size=latent_size, num_heads=num_heads) for _ in range(n_layers)
        ])
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(latent_size, output_size)

    def forward(self, x):
        """
        Forward pass of the ViTNetwork.

        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, depth, height, width].

        Returns:
            Tensor: Output logits of shape [batch_size, output_size].
        """
        y = x
        y = super().forward(y)  # Convert to float16

        y = self.patches(y)  # [batch, latent_size, D', H', W']
        y = y.flatten(2)  # [batch, latent_size, D'*H'*W']
        y = y.permute(0, 2, 1)  # [batch, num_patches, latent_size]
        pos_emb = self.position_embedding(y)  # [1, num_patches, latent_size]
        y = y + pos_emb  # Add positional encodings
        y = self.transformer_blocks(y)  # [batch, num_patches, latent_size]
        y = y.permute(0, 2, 1)  # [batch, latent_size, num_patches]
        y = self.pooling(y).squeeze(-1)  # [batch, latent_size]
        y = self.linear(y)  # [batch, output_size]
        return y

# ----------------------------
# Instantiate DataModule and Check Data Shapes
# ----------------------------


# Define the path to your dataset
dataset_path = "../NPYdata"  # Update this path to your dataset directory

# Instantiate DataModule with torch.uint8 dtype
data_module = Numpy3DDataModule(
    batch_size=8, 
    location=dataset_path, 
    num_workers=0,  # Set to 0 to avoid shared memory issues
    dtype=torch.uint8  # Use torch.uint8 for binary data
)

# Setup the data module (prepare datasets)
data_module.setup('fit')

# Create data loaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

# Check shapes of a single batch
for batch in train_loader:
    x, y = batch
    print(f"Input batch shape: {x.shape}, Label batch shape: {y.shape}")
    break
# Expected Output:
# Input batch shape: torch.Size([8, 1, 128, 128, 128]), Label batch shape: torch.Size([8])

# ----------------------------
# Instantiate and Summarize the Model
# ----------------------------

# Assuming all samples have the same input shape
input_shape = data_module.input_shape  # [1, D, H, W]
output_size = data_module.output_shape  # Number of classes

vit_net = ViTNetwork(
    input_shape=input_shape,  # e.g., [1, 128, 128, 128]
    patch_shape=(16, 16, 16),  # Example patch size for 3D data: (depth, height, width)
    output_size=output_size,  # Number of classes
    latent_size=256,  # Size of latent vector
    n_layers=6,  # Number of transformer layers
    num_heads=4  # Number of attention heads
)

# Use summary to inspect the model
print("\nModel Summary:")
# summary(vit_net, input_size=input_shape)

# ----------------------------
# Setup Logger and Trainer
# ----------------------------

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint

# Initialize the logger
logger = CSVLogger("logs", name="2024-10-10-Transformers",version="vit")

# Define callbacks
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=30,
    verbose=True,
    mode='min'
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min',
)

# Initialize the Trainer
trainer = pl.Trainer(
    accelerator="gpu",
    # devices=[1],  # Directly specifies to use GPU 1
    precision='16-mixed',  # Enable mixed precision
    max_epochs=100,
    logger=logger,
    callbacks=[TQDMProgressBar(refresh_rate=50)],#, early_stop_callback, checkpoint_callback],
    log_every_n_steps=50  # Log every 50 steps
)

# ----------------------------
# Train the Model
# ----------------------------

# Run training
print("\nStarting Training:")
trainer.fit(vit_net, datamodule=data_module)

print("\nStarting Testing:")
test_results = trainer.test(vit_net, datamodule=data_module)

# Display test metrics
test_acc = test_results[0]['test_acc']
test_loss = test_results[0]['test_loss']
test_f1 = test_results[0]['test_f1']

print(f"\nTest Results:")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
