import os
from ultralytics import YOLO
import torch
import argparse

def train_yolo11(data_path, epochs, imgsz, batch, resume=False, checkpoint_path=None):
    """
    Train YOLO11 for segmentation with multiple object classes.

    Args:
        data_path (str): Path to the YOLO .yaml file.
        epochs (int): Number of training epochs.
        imgsz (int): Image size.
        batch (int): Batch size.
        resume (bool): Resume from checkpoint.
        checkpoint_path (str): Path to checkpoint .pt file.
    """
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.exists(data_path):
        print(f"Error: Dataset YAML file not found at {data_path}")
        return None

    # Load model
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        pretrained_weights = "yolo11n-seg.pt"
        print(f"Loading pretrained model {pretrained_weights} for segmentation...")
        model = YOLO(pretrained_weights)

    # Train
    print(f"Training YOLO11 for segmentation using {device}...")
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=8,
        save=True,
    )

    # Save model
    save_dir = os.path.join("bpc", "yolo", "models", "segmentation", "multi_obj")
    os.makedirs(save_dir, exist_ok=True)
    model_name = "yolo11-segmentation-multi_obj.pt"
    final_model_path = os.path.join(save_dir, model_name)
    model.save(final_model_path)

    print(f"Model saved as: {final_model_path}")
    return final_model_path

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 for segmentation.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset YAML file (e.g., data_multi_obj.yaml).")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint .pt file.")
    args = parser.parse_args()

    train_yolo11(
        data_path=args.data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path
    )

if __name__ == "__main__":
    main()