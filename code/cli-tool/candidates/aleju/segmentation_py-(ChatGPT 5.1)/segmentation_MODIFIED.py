"""
Simple pytorch lightning segmentation example
"""

# Imports
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations for our images
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Easier dataset management via mini-batches
from tqdm import tqdm  # For nice progress bar!
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import Callback, EarlyStopping

import numpy as np

precision = "medium"
torch.set_float32_matmul_precision(precision)
criterion = nn.CrossEntropyLoss()  # Per-pixel cross entropy for segmentation


class SimpleUNetLightning(pl.LightningModule):
    def __init__(self, lr=3e-4, in_channels=3, num_classes=21):
        """
        Simple encoder-decoder / U-Net style model for semantic segmentation.

        Args:
            lr: Learning rate for the optimizer.
            in_channels: Number of input channels (3 for RGB).
            num_classes: Number of segmentation classes (21 for VOC2012).
        """
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes

        # Metrics: per-pixel IoU (Jaccard) for train/val/test
        self.train_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes
        )
        self.val_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes
        )
        self.test_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes
        )

        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)  # 256 (up) + 256 (skip)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)  # 128 (up) + 128 (skip)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)  # 64 (up) + 64 (skip)

        # Final classifier: per-pixel logits over classes
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    @staticmethod
    def _conv_block(in_channels, out_channels):
        """Simple 2-layer conv block with ReLU + BatchNorm."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _forward_unet(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        u3 = self.up3(b)
        # Pad if needed (for odd input sizes)
        if u3.shape[-2:] != e3.shape[-2:]:
            diff_y = e3.shape[-2] - u3.shape[-2]
            diff_x = e3.shape[-1] - u3.shape[-1]
            u3 = F.pad(u3, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        if u2.shape[-2:] != e2.shape[-2:]:
            diff_y = e2.shape[-2] - u2.shape[-2]
            diff_x = e2.shape[-1] - u2.shape[-1]
            u2 = F.pad(u2, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            diff_y = e1.shape[-2] - u1.shape[-2]
            diff_x = e1.shape[-1] - u1.shape[-1]
            u1 = F.pad(u1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        logits = self.classifier(d1)
        return logits

    # This mirrors the CNNLightning pattern: separate "common" forward step
    def _common_step(self, x, batch_idx):
        del batch_idx  # Not used, kept to match the signature style
        logits = self._forward_unet(x)
        return logits

    def forward(self, x):
        return self._forward_unet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (N, C, H, W), y: (N, H, W) long
        logits = self._common_step(x, batch_idx)
        loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        iou = self.train_iou(preds, y)

        # Log the metric object like in lightning_simple_CNN
        self.log(
            "train_iou_step",
            self.train_iou,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs):
        # Reset running metric state
        self.train_iou.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self._common_step(x, batch_idx)
        loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        iou = self.val_iou(preds, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self._common_step(x, batch_idx)
        loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        iou = self.test_iou(preds, y)

        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log("test_iou", iou, on_step=True, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self._common_step(x, batch_idx)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class VOCSegmentationDataModule(pl.LightningDataModule):
    """
    DataModule for Pascal VOC 2012 segmentation, similar to MNISTDataModule
    in lightning_simple_CNN.py.
    """

    def __init__(self, batch_size=4, num_workers=4, root="dataset/"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @staticmethod
    def _mask_to_tensor(mask):
        """
        Convert PIL mask to (H, W) long tensor.
        VOC masks are stored as indexed PNGs.
        """
        mask_np = np.array(mask, dtype=np.int64)
        return torch.from_numpy(mask_np)

    def setup(self, stage=None):
        # Training set
        self.train_dataset = datasets.VOCSegmentation(
            root=self.root,
            year="2012",
            image_set="train",
            download=True,
            transform=transforms.ToTensor(),
            target_transform=self._mask_to_tensor,
        )

        # Validation set
        self.val_dataset = datasets.VOCSegmentation(
            root=self.root,
            year="2012",
            image_set="val",
            download=True,
            transform=transforms.ToTensor(),
            target_transform=self._mask_to_tensor,
        )

        # For simplicity, reuse val as test here. Replace with your own split if needed.
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Segmentation training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Segmentation training is ending")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Run training like in lightning_simple_CNN.py
if __name__ == "__main__":
    # Initialize network
    model_lightning = SimpleUNetLightning(
        lr=3e-4,
        in_channels=3,
        num_classes=21,  # VOC2012 has 21 classes (including background)
    )

    dm = VOCSegmentationDataModule(
        batch_size=4,
        num_workers=4,
        root="dataset/",
    )

    trainer = pl.Trainer(
        # fast_dev_run=True,
        # overfit_batches=3,
        max_epochs=20,
        precision=16,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=[0] if torch.cuda.is_available() else None,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            MyPrintingCallback(),
        ],
        auto_lr_find=False,
        enable_model_summary=True,
        profiler="simple",
        # strategy="deepspeed_stage_1",  # Uncomment if you actually want deepspeed
        # accumulate_grad_batches=2,
        # auto_scale_batch_size="binsearch",
        # log_every_n_steps=1,
    )

    # Fit model
    trainer.fit(
        model=model_lightning,
        datamodule=dm,
    )

    # Test model
    trainer.test(model=model_lightning, datamodule=dm)
