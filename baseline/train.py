from pathlib import Path
from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


from baseline.collate import pad_collate
from baseline.dataset import BaselineDataset
from baseline.model import SimpleSegmentationModel

# NEW
from baseline.models.unet.unet_model import UNet
from baseline.utils.cleaning import remove_all_low_ndvi_images


def print_iou_per_class(
    targets: torch.Tensor,
    preds: torch.Tensor,
    nb_classes: int,
) -> None:
    """
    Compute IoU between predictions and targets, for each class.

    Args:
        targets (torch.Tensor): Ground truth of shape (B, H, W).
        preds (torch.Tensor): Model predictions of shape (B, nb_classes, H, W).
        nb_classes (int): Number of classes in the segmentation task.
    """

    # Compute IoU for each class
    # Note: I use this for loop to iterate also
    # on classes not in the demo batch

    iou_per_class = []
    for class_id in range(nb_classes):
        iou = jaccard_score(
            targets == class_id,
            preds == class_id,
            average="binary",
            zero_division=0,
        )
        iou_per_class.append(iou)

    for class_id, iou in enumerate(iou_per_class):
        print(
            "class {} - IoU: {:.4f} - targets: {} - preds: {}".format(
                class_id, iou,
                (targets == class_id).sum(),
                (preds == class_id).sum()
            )
        )


def print_mean_iou(targets: torch.Tensor, preds: torch.Tensor) -> None:
    """
    Compute mean IoU between predictions and targets.

    Args:
        targets (torch.Tensor): Ground truth of shape (B, H, W).
        preds (torch.Tensor): Model predictions of shape (B, nb_classes, H, W).
    """

    mean_iou = jaccard_score(targets, preds, average="macro")
    print(f"meanIOU (over existing classes in targets): {mean_iou:.4f}")


def reduce_4D_to_3D(x: torch.Tensor) -> torch.Tensor:
    """
    Reduce a 4D tensor to 3D by taking the mean over the time dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, H, W).

    Returns:
        torch.Tensor: Output tensor of shape (B, H, W).
    """
    return x.mean(dim=1)


def train_model(
    data_folder: Path,
    nb_classes: int,
    input_channels: int,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    device: str = "cuda",
    verbose: bool = False,
) -> SimpleSegmentationModel:
    """
    Training pipeline.
    """

    # Create data loader
    dataset = BaselineDataset(data_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=pad_collate, shuffle=True
    )

    # Initialize the model, loss function, and optimizer
    model = UNet(n_channels=input_channels, n_classes=nb_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay,
                              momentum=momentum,
                              foreach=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move the model to the appropriate device (GPU if available)
    device = torch.device(device)
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for _, (inputs, targets) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            # NEW - Remove images with low NDVI
            ndvi_threshold = 0.3
            inputs_red = remove_all_low_ndvi_images(
                inputs, ndvi_threshold
            )
            # NEW - Drop the time dimension before feeding the data
            inputs_red_3D = reduce_4D_to_3D(inputs_red)

            # Move data to device
            inputs_red_3D = inputs_red_3D.to(device)  # Satellite data

            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # outputs = model(inputs["S2"][:, 10, :, :, :])  # only 10th img
            outputs = model(inputs_red_3D)

            # Loss computation
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Get the predicted class per pixel (B, H, W)
            preds = torch.argmax(outputs, dim=1)

            # Move data from GPU/Metal to CPU
            targets = targets.cpu().numpy().flatten()
            preds = preds.cpu().numpy().flatten()

            if verbose:
                # Print IOU for debugging
                print_iou_per_class(targets, preds, nb_classes)
                print_mean_iou(targets, preds)

        # Print the loss for this epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete.")
    return model


if __name__ == "__main__":
    # Example usage:
    model = train_model(
        data_folder=Path(
            "/content/drive/MyDrive/hackathon-mines-invent-2024/MINIDATA/TRAIN"
        ),
        time_channels=43,
        nb_classes=20,
        input_channels=10,
        num_epochs=1000,
        batch_size=32,  # 32 for full dataset
        learning_rate=1e-3,
        device="cuda",
        verbose=True,
    )
