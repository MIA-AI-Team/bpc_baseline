import os
import torch
from ultralytics import YOLO

def train_multi_class_yolo(
    yaml_path: str,
    task: str = "detection",      # or "segmentation"
    epochs: int = 300,
    imgsz: int = 640,
    batch: int = 16,
    drive_project: str = "/content/drive/MyDrive/bpc_opencv_dataset/ipd/Yolo/runs",
    run_name:    str = "yolo11-detection-multi"
):
    device = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    weights = "yolo11m.pt" if task == "detection" else "yolo11m-seg.pt"
    print(f"Loading pretrained weights: {weights}")
    os.makedirs(drive_project, exist_ok=True)

    model = YOLO(weights)
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=8,
        project=drive_project,   
        name=run_name,           
        exist_ok=True,           
    )

    run_dir      = os.path.join(drive_project, run_name)
    last_ckpt    = os.path.join(run_dir, "weights", "last.pt")
    best_ckpt    = os.path.join(run_dir, "weights", "best.pt")
    metrics_csv  = os.path.join(run_dir, "metrics.csv")
    results_png  = os.path.join(run_dir, "results.png")

    print(f"\n All logs & checkpoints saved under:\n   {run_dir}")
    print(f"   • last.pt: {last_ckpt}")
    print(f"   • best.pt: {best_ckpt}")
    print(f"   • metrics: {metrics_csv}")
    print(f"   • plots:   {results_png}")

    return last_ckpt

if __name__ == "__main__":

    train_multi_class_yolo(
        yaml_path    = "/content/drive/MyDrive/bpc_opencv_dataset/ipd/Yolo/data_multi.yaml",
        task         = "detection",
        epochs       = 200,
        imgsz        = 640,
        batch        = 64,
        drive_project= "/content/drive/MyDrive/bpc_opencv_dataset/ipd/Yolo/runs",
        run_name     = "yolo11-detection-multi"
    )
