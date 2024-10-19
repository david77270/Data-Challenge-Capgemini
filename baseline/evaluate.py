from pathlib import Path
from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
from tqdm import tqdm

from baseline.collate import pad_collate
from baseline.dataset import BaselineDataset
from baseline.model import SimpleSegmentationModel
from baseline.train import print_iou_per_class, print_mean_iou


def evaluate_model(
    model: SimpleSegmentationModel,
    data_folder: Path,
    nb_classes: int,
    batch_size: int = 4,
    device: str = "cuda",
    verbose: bool = False,
):
    """
    Evaluates the trained model on the test dataset.
    """

    # Set the device
    device = torch.device(device)
    model.to(device)

    # Create data loader for the test set
    test_dataset = BaselineDataset(data_folder)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=pad_collate
    )

    # Set the model to evaluation mode
    model.eval()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Tracking variables
    total_loss = 0.0
    all_preds = []
    all_targets = []

    # Disable gradient calculations for evaluation (reduces memory usage)
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(test_dataloader),
                                         total=len(test_dataloader)):
            # Move data to device
            inputs["S2"] = inputs["S2"].to(device)  # Satellite data
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs["S2"])

            # Loss computation
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Get predicted class per pixel
            preds = torch.argmax(outputs, dim=1)

            # Move data from GPU to CPU
            preds = preds.cpu().numpy().flatten()
            targets = targets.cpu().numpy().flatten()

            # Collect predictions and targets for metrics
            all_preds.extend(preds)
            all_targets.extend(targets)

            if verbose:
                # Print per-class IOU for debugging
                print_iou_per_class(targets, preds, nb_classes)
                print_mean_iou(targets, preds)

    # Compute the overall metrics
    mean_loss = total_loss / len(test_dataloader)

    # Compute the Mean IOU for all predictions
    mean_iou = jaccard_score(all_targets, all_preds, average="macro",
                             labels=list(range(nb_classes)))

    # Print results
    print(f"Evaluation Loss: {mean_loss:.4f}")
    print(f"Mean IOU: {mean_iou:.4f}")

    return mean_loss, mean_iou
