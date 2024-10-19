import torch
from tqdm import tqdm
from pathlib import Path
from baseline.collate import pad_collate
from baseline.dataset import TestDataset
from baseline.model import SimpleSegmentationModel
from baseline.train import reduce_4D_to_3D
from baseline.utils.cleaning import remove_all_low_ndvi_images
from baseline.submission_tools import masks_to_str


def predict_model(
    model: SimpleSegmentationModel,
    data_folder: Path,
    batch_size: int = 4,
    device: str = "cuda",
    verbose: bool = False,
) -> list[str]:
    """
    Generates predictions from the model on the test dataset.

    Args:
        model (SimpleSegmentationModel): Trained segmentation model.
        data_folder (Path): Path to the folder containing the test dataset.
        batch_size (int, optional): Batch size for data loading. Defaults to 4.
        device (str, optional): Device for computation (e.g., 'cuda' or 'cpu').
            Defaults to 'cuda'.
        verbose (bool, optional): Whether to print debug information.
            Defaults to False.

    Returns:
        list[str]: A list of stringified prediction masks for
        each image in the dataset.
    """

    # Set the device
    device = torch.device(device)
    model.to(device)

    # Create data loader for the test set
    test_dataset = TestDataset(data_folder)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=pad_collate
    )

    # Set the model to evaluation mode
    model.eval()

    # List to store the stringified predictions
    stringified_predictions = []

    # Disable gradient calculations for evaluation (reduces memory usage)
    # Disable gradient calculations for evaluation (reduces memory usage)
    with torch.no_grad():
        for _, (inputs, _) in tqdm(enumerate(test_dataloader),
                                   total=len(test_dataloader)):
            # NEW - Remove images with low NDVI
            ndvi_threshold = 0.3
            inputs_red = remove_all_low_ndvi_images(
                inputs, ndvi_threshold
            )
            # NEW - Drop the time dimension before feeding the data
            inputs_red_3D = reduce_4D_to_3D(inputs_red)

            # Move data to device
            inputs_red_3D = inputs_red_3D.to(device)  # Satellite data

            # Forward pass
            outputs = model(inputs_red_3D)

            # Get predicted class per pixel
            preds = torch.argmax(outputs, dim=1)  # Shape (B, H, W)

            # Move predictions to CPU and convert to numpy
            preds_np = preds.cpu().numpy()

            # Convert predictions to string format using masks_to_str function
            stringified_batch_preds = masks_to_str(preds_np)
            stringified_predictions.extend(stringified_batch_preds)

            if verbose:
                print(f"Processed batch of {len(preds_np)} images.")

    return stringified_predictions
