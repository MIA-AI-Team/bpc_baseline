import os
import json
import shutil
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

def prepare_train_pbr(train_pbr_path, train_pbr_enhanced_path, output_path, obj_ids):
    """
    Prepare the train_pbr dataset for YOLO segmentation, processing all specified object IDs.
    Creates 'images' and 'labels' folders under output_path with segmentation annotations.
    """
    # Cameras to scan
    cameras = ["rgb_cam1", "rgb_cam2", "rgb_cam3"]

    # Corresponding ground-truth and mask folders
    camera_gt_map = {
        "rgb_cam1": "scene_gt_cam1.json",
        "rgb_cam2": "scene_gt_cam2.json",
        "rgb_cam3": "scene_gt_cam3.json"
    }
    camera_gt_info_map = {
        "rgb_cam1": "scene_gt_info_cam1.json",
        "rgb_cam2": "scene_gt_info_cam2.json",
        "rgb_cam3": "scene_gt_info_cam3.json"
    }
    mask_folders = {
        "rgb_cam1": "mask_cam1",
        "rgb_cam2": "mask_cam2",
        "rgb_cam3": "mask_cam3"
    }

    # Map obj_ids to class indices (0 to len(obj_ids)-1)
    obj_id_to_class = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}

    # Ensure output directories exist
    images_dir = os.path.join(output_path, "images")
    labels_dir = os.path.join(output_path, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Iterate over scenes (000000 to 000049)
    scene_folders = [
        d for d in os.listdir(train_pbr_enhanced_path)
        if os.path.isdir(os.path.join(train_pbr_enhanced_path, d)) and not d.startswith(".")
    ]
    scene_folders.sort()

    for scene_folder in tqdm(scene_folders, desc="Processing train_pbr scenes"):
        scene_path = os.path.join(train_pbr_enhanced_path, scene_folder)
        scene_path_gt = os.path.join(train_pbr_path, scene_folder)  # For GT and masks

        for cam in cameras:
            rgb_path = os.path.join(scene_path, cam)
            mask_path = os.path.join(scene_path_gt, mask_folders[cam])
            scene_gt_file = os.path.join(scene_path_gt, camera_gt_map[cam])
            scene_gt_info_file = os.path.join(scene_path_gt, camera_gt_info_map[cam])

            if not os.path.exists(rgb_path):
                print(f"Missing RGB folder for {cam} in {scene_folder}: {rgb_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"Missing mask folder for {cam} in {scene_folder}: {mask_path}")
                continue
            if not os.path.exists(scene_gt_file):
                print(f"Missing JSON file for {cam} in {scene_folder}: {scene_gt_file}")
                continue
            if not os.path.exists(scene_gt_info_file):
                print(f"Missing JSON file for {cam} in {scene_folder}: {scene_gt_info_file}")
                continue

            # Load JSON files
            with open(scene_gt_file, "r") as f:
                scene_gt_data = json.load(f)
            with open(scene_gt_info_file, "r") as f:
                scene_gt_info_data = json.load(f)

            num_imgs = len(scene_gt_data)
            for img_id in range(num_imgs):
                img_key = str(img_id)
                img_file_jpg = os.path.join(rgb_path, f"{img_id:06d}.jpg")
                img_file_png = os.path.join(rgb_path, f"{img_id:06d}.png")
                img_file = img_file_jpg if os.path.exists(img_file_jpg) else img_file_png if os.path.exists(img_file_png) else None

                if img_file is None:
                    continue

                if img_key not in scene_gt_data or img_key not in scene_gt_info_data:
                    continue

                # Get image dimensions
                with Image.open(img_file) as img:
                    img_width, img_height = img.size

                # Process objects in the image
                annotations = []
                for idx, (bbox_info, gt_info) in enumerate(zip(scene_gt_info_data[img_key], scene_gt_data[img_key])):
                    obj_id = gt_info["obj_id"]
                    if obj_id not in obj_ids or bbox_info["visib_fract"] <= 0:
                        continue

                    # Load mask
                    mask_file = os.path.join(mask_path, f"{img_id:06d}_{idx:06d}.png")
                    if not os.path.exists(mask_file):
                        continue

                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue

                    # Extract contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue

                    # Convert largest contour to YOLO format
                    contour = max(contours, key=cv2.contourArea)
                    contour = contour.squeeze()
                    if len(contour) < 3:
                        continue

                    # Normalize coordinates
                    points = []
                    for pt in contour:
                        x, y = pt
                        points.extend([x / img_width, y / img_height])

                    # YOLO segmentation format: class x1 y1 x2 y2 ... xn yn
                    class_id = obj_id_to_class[obj_id]
                    annotation = f"{class_id} {' '.join(f'{p:.6f}' for p in points)}"
                    annotations.append(annotation)

                if not annotations:
                    continue

                # Copy image
                out_img_name = f"{scene_folder}_{cam}_{img_id:06d}.jpg"
                out_img_path = os.path.join(images_dir, out_img_name)
                shutil.copy(img_file, out_img_path)

                # Write label file
                out_label_name = f"{scene_folder}_{cam}_{img_id:06d}.txt"
                out_label_path = os.path.join(labels_dir, out_label_name)
                with open(out_label_path, "w") as lf:
                    for annotation in annotations:
                        lf.write(f"{annotation}\n")

def generate_yaml(output_path, obj_ids):
    """
    Generate a YOLO .yaml file for segmentation with multiple objects.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_configs_dir = os.path.join(script_dir, "configs")
    os.makedirs(yolo_configs_dir, exist_ok=True)

    images_dir = os.path.join(output_path, "images")
    train_path = os.path.abspath(images_dir)
    val_path = os.path.abspath(images_dir)

    yaml_path = os.path.join(yolo_configs_dir, "data_multi_obj.yaml")

    yaml_content = {
        "train": train_path,
        "val": val_path,
        "nc": len(obj_ids),
        "names": [f"object_{obj_id}" for obj_id in sorted(obj_ids)]
    }

    with open(yaml_path, "w") as f:
        for key, value in yaml_content.items():
            if key in ["train", "val"]:
                f.write(f"{key}: {value}\n")
            elif key == "nc":
                f.write(f"{key}: {value}\n")
            elif key == "names":
                f.write(f"{key}: {yaml_content['names']}\n")

    print(f"[INFO] Generated YAML file at: {yaml_path}\n")
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Prepare train_pbr dataset for YOLO segmentation.")
    parser.add_argument("--train_pbr_path", type=str, required=True,
                        help="Path to train_pbr dataset.")
    parser.add_argument("--train_pbr_enhanced_path", type=str, required=True,
                        help="Path to train_pbr_enhanced dataset.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for YOLO dataset.")
    args = parser.parse_args()

    obj_ids = [0, 8, 18, 19, 20, 1, 4, 10, 11, 14]

    # Prepare data
    prepare_train_pbr(args.train_pbr_path, args.train_pbr_enhanced_path, args.output_path, obj_ids)

    # Generate YAML
    generate_yaml(args.output_path, obj_ids)

    print("[INFO] Dataset preparation complete!")

if __name__ == "__main__":
    main()
