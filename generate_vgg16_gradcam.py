import argparse
import csv
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models import VGG16_Weights


os.environ.setdefault("TORCH_HOME", "/data/ramialle/research_training/Research_Training/.torch")

CLASS_NAMES = {0: "malignant", 1: "benign"}
IMG_SIZE = (224, 224)
DEFAULT_DATASET_DIR = Path("/data/ramialle/datasets2/Dataset_BUSI_with_GT_Clean")
DEFAULT_IMAGE_DIR = DEFAULT_DATASET_DIR / "images-20260602T203626Z-3-001" / "images"
DEFAULT_CHECKPOINT = Path("model_checkpoints/vgg16_best.pth")
DEFAULT_OUTPUT_DIR = Path("vgg16_gradcam_outputs")


def column_index(cell_reference):
    letters = "".join(ch for ch in cell_reference if ch.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def read_xlsx_rows(path, image_dir):
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with ZipFile(path) as workbook_zip:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook_zip.namelist():
            root = ET.fromstring(workbook_zip.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", namespace):
                text_parts = [node.text or "" for node in item.findall(".//main:t", namespace)]
                shared_strings.append("".join(text_parts))

        sheet = ET.fromstring(workbook_zip.read("xl/worksheets/sheet1.xml"))
        raw_rows = []
        for row in sheet.findall(".//main:sheetData/main:row", namespace):
            values = []
            for cell in row.findall("main:c", namespace):
                column = column_index(cell.attrib.get("r", "A1"))
                while len(values) <= column:
                    values.append("")

                cell_type = cell.attrib.get("t")
                value_node = cell.find("main:v", namespace)
                inline_node = cell.find("main:is/main:t", namespace)

                if cell_type == "s" and value_node is not None:
                    value = shared_strings[int(value_node.text)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is not None:
                    value = value_node.text or ""
                else:
                    value = ""

                values[column] = value
            raw_rows.append(values)

    header = [str(value).strip() for value in raw_rows[0]]
    rows = []
    for row in raw_rows[1:]:
        row = row + [""] * (len(header) - len(row))
        record = {key: value for key, value in zip(header, row)}
        if record.get("Image"):
            record["Image"] = str(record["Image"]).strip()
            record["Label"] = int(float(record["Label"]))
            record["path"] = image_dir / record["Image"]
            record["class_name"] = CLASS_NAMES[record["Label"]]
            rows.append(record)
    return rows


def load_rows(split, dataset_dir, image_dir):
    split_files = {
        "train": [dataset_dir / "train.xlsx"],
        "test": [dataset_dir / "test.xlsx"],
        "all": [dataset_dir / "train.xlsx", dataset_dir / "test.xlsx"],
    }
    rows = []
    for split_file in split_files[split]:
        split_name = split_file.stem
        for row in read_xlsx_rows(split_file, image_dir):
            row["split"] = split_name
            rows.append(row)
    missing = [str(row["path"]) for row in rows if not row["path"].exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} image(s), first missing path: {missing[0]}")
    return rows


def build_vgg16():
    weights = VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.15),
        nn.Linear(256, 2),
    )
    return model, weights


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_model(checkpoint_path, device):
    model, weights = build_vgg16()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint))
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    model.to(device)
    model.eval()
    return model, weights


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(torch.argmax(probabilities, dim=1).item())
        if target_class is None:
            target_class = predicted_class

        score = logits[:, target_class].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=IMG_SIZE, mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return {
            "cam": cam,
            "predicted_class": predicted_class,
            "predicted_probability": float(probabilities[0, predicted_class].item()),
            "target_class": int(target_class),
            "target_probability": float(probabilities[0, target_class].item()),
        }


def colorize_cam(cam):
    # Full-color importance map: blue/cyan/green/yellow/red from least to most important.
    red = np.clip(1.5 * cam - 0.5, 0, 1)
    green = np.clip(1.5 - np.abs(3.0 * cam - 1.5), 0, 1)
    blue = np.clip(1.0 - 1.5 * cam, 0, 1)
    heatmap = np.stack([red, green, blue], axis=-1)
    return (heatmap * 255).astype(np.uint8)


def sanitize_filename(name):
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")


def save_cam_images(row, cam, output_dir, alpha):
    original = Image.open(row["path"]).convert("RGB")
    original_resized = original.resize(IMG_SIZE)
    original_array = np.asarray(original_resized).astype(np.float32)
    heatmap_array = colorize_cam(cam).astype(np.float32)
    overlay_array = ((1.0 - alpha) * original_array + alpha * heatmap_array).clip(0, 255).astype(np.uint8)

    base_name = sanitize_filename(row["Image"])
    class_dir = output_dir / row["split"] / row["class_name"]
    class_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = class_dir / f"{base_name}_heatmap.png"
    overlay_path = class_dir / f"{base_name}_gradcam.png"

    Image.fromarray(heatmap_array.astype(np.uint8)).save(heatmap_path)
    Image.fromarray(overlay_array).save(overlay_path)
    return heatmap_path, overlay_path


def write_summary(summary_rows, output_dir):
    summary_path = output_dir / "cam_summary.csv"
    fieldnames = [
        "split",
        "image",
        "true_label",
        "true_class",
        "predicted_label",
        "predicted_class",
        "predicted_probability",
        "cam_target_label",
        "cam_target_class",
        "cam_target_probability",
        "heatmap_path",
        "overlay_path",
    ]
    with summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for the trained VGG16 breast tumor classifier."
    )
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", choices=["predicted", "true"], default="predicted")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    rows = load_rows(args.split, args.dataset_dir, args.image_dir)
    model, weights = load_model(args.checkpoint, device)
    normalize = weights.transforms()
    preprocess = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize.mean, std=normalize.std),
        ]
    )

    grad_cam = GradCAM(model, model.features[28])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    try:
        for index, row in enumerate(rows, start=1):
            image = Image.open(row["path"]).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device)
            target_class = row["Label"] if args.target == "true" else None

            result = grad_cam.generate(input_tensor, target_class=target_class)
            heatmap_path, overlay_path = save_cam_images(row, result["cam"], args.output_dir, args.alpha)

            summary_rows.append(
                {
                    "split": row["split"],
                    "image": row["Image"],
                    "true_label": row["Label"],
                    "true_class": row["class_name"],
                    "predicted_label": result["predicted_class"],
                    "predicted_class": CLASS_NAMES[result["predicted_class"]],
                    "predicted_probability": f"{result['predicted_probability']:.6f}",
                    "cam_target_label": result["target_class"],
                    "cam_target_class": CLASS_NAMES[result["target_class"]],
                    "cam_target_probability": f"{result['target_probability']:.6f}",
                    "heatmap_path": str(heatmap_path),
                    "overlay_path": str(overlay_path),
                }
            )

            if index % 25 == 0 or index == len(rows):
                print(f"Generated CAMs for {index}/{len(rows)} images")
    finally:
        grad_cam.close()

    summary_path = write_summary(summary_rows, args.output_dir)
    print(f"Saved CAM images under: {args.output_dir}")
    print(f"Saved summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()
