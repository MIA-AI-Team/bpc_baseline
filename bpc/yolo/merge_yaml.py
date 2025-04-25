import yaml
from pathlib import Path
import argparse

def merge_yamls(yaml_dir: str, out_yaml: str):
    """
    Merges all per-object YAMLs in yaml_dir into one dataset spec,
    and remaps labels so class IDs align.
    """
    yaml_paths = list(Path(yaml_dir).glob("*.yaml"))
    # 1. Gather all unique class names
    class_names = []
    for p in yaml_paths:
        d = yaml.safe_load(p.read_text())
        for idx, name in d["names"].items():
            if name not in class_names:
                class_names.append(name)

    # 2. Build combined spec
    combined = {
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
        "train": [],
        "val": []
    }

    # 3. Collect image paths and remap label files
    for p in yaml_paths:
        d = yaml.safe_load(p.read_text())
        # extend image lists
        combined["train"].extend(d["train"])
        combined["val"].extend(d["val"])

        # remap every label file in train/val folders
        for split in ("train", "val"):
            for img_path in d[split]:
                lbl_path = Path(img_path).with_suffix(".txt").parent.parent / "labels" / Path(img_path).stem + ".txt"
                if not lbl_path.exists(): 
                    continue
                # read, remap, overwrite
                lines = lbl_path.read_text().splitlines()
                new_lines = []
                for line in lines:
                    parts = line.split()
                    old_class = int(parts[0])
                    old_name  = d["names"][old_class]
                    new_class = class_names.index(old_name)
                    new_lines.append(" ".join([str(new_class)] + parts[1:]))
                lbl_path.write_text("\n".join(new_lines))

    # 4. Dump merged YAML
    Path(out_yaml).write_text(yaml.dump(combined))
    print(f"Merged {len(yaml_paths)} yamls → {out_yaml} with nc={combined['nc']}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple per-object YOLO .yaml specs into one dataset.yaml"
    )
    parser.add_argument(
        "yaml_dir",
        help="Directory containing one .yaml file per object"
    )
    parser.add_argument(
        "out_yaml",
        help="Output path for the merged dataset YAML"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    merge_yamls(args.yaml_dir, args.out_yaml)
