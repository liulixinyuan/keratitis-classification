#!/usr/bin/env python3
"""
Post-training evaluation with bootstrap 95% confidence intervals.

This script supports two common layouts in this project:
1. single holdout weights:
   checkpoints/.../{Model}/best_model_{Model}.pth
2. patient-level 5-fold weights:
   checkpoints/.../{Model}/fold_{i}/best_model_{Model}_fold{i}.pth

For 5-fold evaluation it rebuilds the same patient-level validation folds from
train_transformed, runs each fold model on its own validation fold, concatenates
all out-of-fold predictions, and then bootstraps the final metrics.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_MAPPING = {
    "NC": 0,
    "BK": 1,
    "FK": 2,
    "VK": 3,
    "NIFK": 4,
    "AMK": 5,
}
NEW_CLASS_NAMES = ["NC", "BK", "FK", "VK", "NIFK", "AMK"]
NUM_CLASSES = 6
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
FOUNDATION_MODEL_NAMES = ["DINOv3_base", "CTransPath", "clip_prompt", "DINOV2"]
FOUNDATION_PROTOCOLS = ("zero_shot", "linear_probe", "full_finetune", "partial_finetune", "few_shot")
OUTPUT_MODEL_ORDER = [
    "InceptionV3",
    "ResNet18",
    "ResNet50",
    "DenseNet121",
    "MobileNetV2",
    "ResNeXt50",
    "RegNet",
    "EfficientNet_B0",
    "ConvNeXT",
    "ViT",
    "DINOV2_linear_probe",
    "DINOV2_full_finetune",
    "DINOv3_base_linear_probe",
    "DINOv3_base_full_finetune",
    "CTransPath_linear_probe",
    "CTransPath_full_finetune",
    "clip_prompt_zero_shot",
    "clip_prompt_linear_probe",
    "clip_prompt_full_finetune",
]
OUTPUT_MODEL_ORDER_ALIASES = {
    "DINOv3_linear_probe": "DINOv3_base_linear_probe",
    "DINOv3_full_finetune": "DINOv3_base_full_finetune",
}
OUTPUT_MODEL_ORDER_INDEX = {model_name: idx for idx, model_name in enumerate(OUTPUT_MODEL_ORDER)}


def load_optional_training_classes():
    """Import heavier project classes only for models that need them."""
    from train import (
        CLIPPromptClassifier,
        CustomClassifier,
        DEFAULT_CLIP_PROMPT_TEMPLATES,
    )
    CLIPClassifier = None
    return CLIPClassifier, CLIPPromptClassifier, CustomClassifier, DEFAULT_CLIP_PROMPT_TEMPLATES


def load_dinov3_class():
    """Import DINOv3 only when that model is actually evaluated."""
    from train import PortableDINOv3Model
    return PortableDINOv3Model


def split_model_name(model_name):
    for base_name in sorted(FOUNDATION_MODEL_NAMES, key=len, reverse=True):
        if model_name == base_name:
            return base_name, "full_finetune"
        prefix = f"{base_name}_"
        if model_name.startswith(prefix):
            suffix = model_name[len(prefix):]
            protocol = next(
                (candidate for candidate in FOUNDATION_PROTOCOLS if suffix.startswith(candidate)),
                "full_finetune",
            )
            return base_name, protocol
    return model_name, "full_finetune"


def discover_model_experiments(base_ckpt_dir, model_name):
    exact_dir = os.path.join(base_ckpt_dir, model_name)
    if os.path.isdir(exact_dir):
        return [model_name]

    base_model, _ = split_model_name(model_name)
    if base_model not in FOUNDATION_MODEL_NAMES:
        return [model_name]

    discovered = []
    if os.path.isdir(base_ckpt_dir):
        for dirname in sorted(os.listdir(base_ckpt_dir)):
            full_path = os.path.join(base_ckpt_dir, dirname)
            if os.path.isdir(full_path) and split_model_name(dirname)[0] == base_model:
                discovered.append(dirname)
    return discovered or [model_name]


class CustomImageFolder(datasets.ImageFolder):
    """ImageFolder with labels remapped to the project's fixed 6-class order."""

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        original_classes = self.classes
        self.samples = [
            (path, CLASS_MAPPING[original_classes[label]])
            for path, label in self.samples
            if original_classes[label] in CLASS_MAPPING
        ]
        self.targets = [label for _, label in self.samples]
        self.classes = NEW_CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}


class PatientImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = list(file_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def build_patient_data(root_dir):
    """Build patient records from train_transformed; augmented images stay out of validation."""
    patient_data = {}
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir) or class_name not in CLASS_MAPPING:
            continue
        label = CLASS_MAPPING[class_name]
        for filename in os.listdir(class_dir):
            if not filename.lower().endswith(IMAGE_EXTS):
                continue
            patient_id = filename.split("_")[0]
            path = os.path.join(class_dir, filename)
            patient_data.setdefault(patient_id, {"label": label, "original": [], "augmented": []})
            if "_aug_" in filename:
                patient_data[patient_id]["augmented"].append(path)
            else:
                patient_data[patient_id]["original"].append(path)
    return patient_data


def patient_balanced_kfold(patient_data, n_splits=5, random_state=42):
    """Same greedy patient-level fold splitter used by the 5-fold training script."""
    rng = np.random.default_rng(random_state)
    all_patient_ids = list(patient_data.keys())
    patients_by_class = defaultdict(list)

    for pid, info in patient_data.items():
        original_count = len(info["original"])
        if original_count == 0:
            raise ValueError(f"Patient {pid} has no original image for validation.")
        patients_by_class[info["label"]].append((pid, original_count))

    fold_patient_ids = [[] for _ in range(n_splits)]
    fold_total_original_counts = [0 for _ in range(n_splits)]
    fold_class_original_counts = [[0 for _ in range(NUM_CLASSES)] for _ in range(n_splits)]

    for label in range(NUM_CLASSES):
        class_patients = patients_by_class[label]
        if len(class_patients) < n_splits:
            raise ValueError(
                f"Class {NEW_CLASS_NAMES[label]} has only {len(class_patients)} patients, "
                f"less than n_splits={n_splits}."
            )

        shuffled_indices = rng.permutation(len(class_patients))
        class_patients = [class_patients[i] for i in shuffled_indices]
        class_patients.sort(key=lambda item: item[1], reverse=True)

        for pid, original_count in class_patients:
            best_fold = min(
                range(n_splits),
                key=lambda fold_idx: (
                    fold_class_original_counts[fold_idx][label],
                    fold_total_original_counts[fold_idx],
                    len(fold_patient_ids[fold_idx]),
                ),
            )
            fold_patient_ids[best_fold].append(pid)
            fold_class_original_counts[best_fold][label] += original_count
            fold_total_original_counts[best_fold] += original_count

    all_patient_id_set = set(all_patient_ids)
    for val_pids in fold_patient_ids:
        val_pid_set = set(val_pids)
        train_pids = [pid for pid in all_patient_ids if pid not in val_pid_set]
        if len(train_pids) + len(val_pids) != len(all_patient_id_set):
            raise RuntimeError("Patient fold split validation failed.")
        yield train_pids, val_pids


def build_fold_val_dataset(patient_data, val_pids, transform):
    val_paths, val_labels = [], []
    for pid in val_pids:
        paths = patient_data[pid]["original"]
        val_paths.extend(paths)
        val_labels.extend([patient_data[pid]["label"]] * len(paths))
    return PatientImageDataset(val_paths, val_labels, transform=transform)


def get_model(model_name, num_classes, class_names=None, clip_freeze=False, clip_hidden=512, device_ids=None):
    """Create the same architecture as training. Final weights are loaded afterward."""
    device_ids = device_ids or [0]
    clip_path = str(PROJECT_ROOT / "weights" / "clip-vit-base-patch32")
    base_model_name, protocol = split_model_name(model_name)

    if base_model_name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif base_model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif base_model_name == "DenseNet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif base_model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif base_model_name == "InceptionV3":
        model = models.inception_v3(weights=None, aux_logits=False, init_weights=False)
        model.AuxLogits = None
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif base_model_name == "ResNeXt50":
        model = models.resnext50_32x4d(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif base_model_name == "RegNet":
        model = models.regnet_x_16gf(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif base_model_name == "EfficientNet_B0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif base_model_name == "ConvNeXT":
        model = models.convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif base_model_name == "ViT":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif base_model_name == "DINOV2":
        model = timm.create_model("vit_small_patch14_dinov2", pretrained=False)
        model.head = nn.Linear(model.num_features, num_classes)
    elif base_model_name == "DINOv3_base":
        DINOv3ModelParallel = load_dinov3_class()
        dino_adaptation = "full_finetune" if protocol == "few_shot" else protocol
        return DINOv3ModelParallel(num_classes, adaptation=dino_adaptation)
    elif base_model_name == "clip_head":
        CLIPClassifier, _, _, _ = load_optional_training_classes()
        if CLIPClassifier is None:
            raise ValueError("clip_head requires CLIPClassifier from train_asoct_6classes.py; use clip_prompt for VIT experiments.")
        return CLIPClassifier(
            num_classes=num_classes,
            clip_path=clip_path,
            freeze_backbone=clip_freeze,
            hidden_dim=clip_hidden,
        )
    elif base_model_name == "clip_prompt":
        _, CLIPPromptClassifier, _, prompt_templates = load_optional_training_classes()
        return CLIPPromptClassifier(
            class_names=class_names,
            clip_path=clip_path,
            prompt_templates=prompt_templates,
            protocol=protocol,
        )
    elif base_model_name == "CTransPath":
        _, _, CustomClassifier, _ = load_optional_training_classes()
        backbone = timm.create_model(
            "vit_small_patch16_224",
            img_size=224,
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            num_classes=0,
            pretrained=False,
        )
        model = CustomClassifier(backbone, in_dim=384, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def resolve_transform(model_name, dinov2_img_size=518):
    base_model_name, _ = split_model_name(model_name)
    if base_model_name in {"DINOV2", "DINOv3_base"}:
        size = dinov2_img_size
    elif base_model_name == "InceptionV3":
        size = 299
    else:
        size = 224

    if base_model_name in {"clip_head", "clip_prompt"}:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def normalize_state_dict_keys(state_dict, model):
    model_keys = model.state_dict().keys()
    has_module_in_model = any(key.startswith("module.") for key in model_keys)
    has_module_in_ckpt = any(key.startswith("module.") for key in state_dict.keys())

    if has_module_in_ckpt and not has_module_in_model:
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    if has_module_in_model and not has_module_in_ckpt:
        return {f"module.{key}": value for key, value in state_dict.items()}
    return state_dict


def load_model_weights(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint), model)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[WARN] Non-strict load for {weights_path}")
        print(f"       missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"       original error: {exc}")
    return model


def compute_sensitivity_specificity(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sensitivities, specificities = [], []
    for idx in range(num_classes):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivities.append(tp / (tp + fn) if (tp + fn) else 0.0)
        specificities.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return np.array(sensitivities), np.array(specificities)


def safe_auc(y_true, y_prob, average="weighted"):
    try:
        return roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
    except ValueError:
        return np.nan


def safe_class_auc(y_true, y_prob, class_idx):
    try:
        return roc_auc_score((y_true == class_idx).astype(int), y_prob[:, class_idx])
    except ValueError:
        return np.nan


def compute_ece(y_true, y_prob, n_bins=15):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        if idx == 0:
            in_bin = (confidences >= bin_boundaries[idx]) & (confidences <= bin_boundaries[idx + 1])
        else:
            in_bin = (confidences > bin_boundaries[idx]) & (confidences <= bin_boundaries[idx + 1])
        if np.any(in_bin):
            ece += np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin])) * np.mean(in_bin)
    return float(ece)


def compute_brier_score(y_true, y_prob, num_classes):
    y_true_onehot = np.eye(num_classes, dtype=float)[y_true]
    return float(np.mean(np.mean((y_true_onehot - y_prob) ** 2, axis=1)))


def compute_overall_metrics(y_true, y_pred, y_prob, num_classes, auc_average="macro"):
    sensitivities, specificities = compute_sensitivity_specificity(y_true, y_pred, num_classes)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": float(np.mean(sensitivities)),
        "specificity": float(np.mean(specificities)),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "auc": safe_auc(y_true, y_prob, average=auc_average),
        "ece": compute_ece(y_true, y_prob),
        "brier": compute_brier_score(y_true, y_prob, num_classes),
    }


def compute_image_metrics(y_true, y_pred, y_prob, y_calibrated_prob, num_classes):
    metrics = compute_overall_metrics(y_true, y_pred, y_prob, num_classes)
    raw_ece = metrics.pop("ece")
    raw_brier = metrics.pop("brier")
    metrics["raw_ece"] = raw_ece
    metrics["raw_brier"] = raw_brier
    metrics["calibrated_ece"] = compute_ece(y_true, y_calibrated_prob)
    metrics["calibrated_brier"] = compute_brier_score(y_true, y_calibrated_prob, num_classes)
    return metrics


def compute_patient_metrics(y_true, y_pred, y_prob, num_classes):
    sensitivities, specificities = compute_sensitivity_specificity(y_true, y_pred, num_classes)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": float(np.mean(sensitivities)),
        "specificity": float(np.mean(specificities)),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "macro_auc": safe_auc(y_true, y_prob, average="macro"),
    }


def compute_per_class_metrics(y_true, y_pred, y_prob, num_classes):
    sensitivities, specificities = compute_sensitivity_specificity(y_true, y_pred, num_classes)
    precision = precision_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    support = np.array([(y_true == idx).sum() for idx in range(num_classes)], dtype=float)
    return {
        idx: {
            "pre": float(precision[idx]),
            "sens": float(sensitivities[idx]),
            "spec": float(specificities[idx]),
            "f1": float(f1[idx]),
            "auc": safe_class_auc(y_true, y_prob, idx),
            "support": float(support[idx]),
        }
        for idx in range(num_classes)
    }


def bootstrap_overall_metrics(y_true, y_pred, y_prob, num_classes, iters, seed=42, metric_fn=None):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metric_fn = metric_fn or compute_overall_metrics
    first = metric_fn(y_true, y_pred, y_prob, num_classes)
    samples = {key: [] for key in first}

    for _ in range(iters):
        idx = rng.integers(0, n, n)
        metrics = metric_fn(y_true[idx], y_pred[idx], y_prob[idx], num_classes)
        for key, value in metrics.items():
            samples[key].append(value)

    return {key: np.asarray(value, dtype=float) for key, value in samples.items()}


def bootstrap_indexed_metrics(n, iters, seed, metric_fn):
    rng = np.random.default_rng(seed)
    first = metric_fn(np.arange(n))
    samples = {key: [] for key in first}

    for _ in range(iters):
        idx = rng.integers(0, n, n)
        metrics = metric_fn(idx)
        for key, value in metrics.items():
            samples[key].append(value)

    return {key: np.asarray(value, dtype=float) for key, value in samples.items()}


def bootstrap_per_class_metrics(y_true, y_pred, y_prob, num_classes, iters, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = {
        cls_idx: {metric: [] for metric in ("pre", "sens", "spec", "f1", "auc", "support")}
        for cls_idx in range(num_classes)
    }

    for _ in range(iters):
        idx = rng.integers(0, n, n)
        metrics = compute_per_class_metrics(y_true[idx], y_pred[idx], y_prob[idx], num_classes)
        for cls_idx, cls_metrics in metrics.items():
            for metric, value in cls_metrics.items():
                samples[cls_idx][metric].append(value)

    return {
        cls_idx: {metric: np.asarray(values, dtype=float) for metric, values in cls_samples.items()}
        for cls_idx, cls_samples in samples.items()
    }


def ci_from_samples(samples, alpha=0.05):
    samples = np.asarray(samples, dtype=float)
    if np.all(np.isnan(samples)):
        return np.nan, np.nan
    return (
        float(np.nanpercentile(samples, 100 * alpha / 2)),
        float(np.nanpercentile(samples, 100 * (1 - alpha / 2))),
    )


def evaluate_model_on_dataset(model_name, weights_path, dataset, batch_size, device, num_classes, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = get_model(model_name, num_classes, class_names=NEW_CLASS_NAMES)
    model = load_model_weights(model, weights_path, device)
    model.to(device)
    model.eval()

    y_true, y_pred, y_prob, y_logits = [], [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            y_prob.append(probs.cpu().numpy())
            y_logits.append(logits.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
            y_true.append(np.asarray(labels))

    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob),
        np.concatenate(y_logits),
        get_dataset_paths(dataset),
    )


def get_dataset_paths(dataset):
    if hasattr(dataset, "file_paths"):
        return list(dataset.file_paths)
    if hasattr(dataset, "samples"):
        return [path for path, _ in dataset.samples]
    return [str(idx) for idx in range(len(dataset))]


def infer_patient_ids(paths):
    return [Path(path).stem.split("_")[0] for path in paths]


def aggregate_patient_predictions(paths, y_true, y_prob):
    grouped = defaultdict(lambda: {"labels": [], "probs": []})
    for patient_id, label, prob in zip(infer_patient_ids(paths), y_true, y_prob):
        grouped[patient_id]["labels"].append(int(label))
        grouped[patient_id]["probs"].append(prob)

    patient_true, patient_prob = [], []
    for patient_id in sorted(grouped):
        labels = grouped[patient_id]["labels"]
        probs = grouped[patient_id]["probs"]
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        patient_true.append(int(np.argmax(counts)))
        patient_prob.append(np.mean(probs, axis=0))

    patient_prob = np.asarray(patient_prob)
    patient_pred = np.argmax(patient_prob, axis=1)
    return np.asarray(patient_true), patient_pred, patient_prob


def add_ci_to_metrics(point, boot):
    ci = {metric: ci_from_samples(values) for metric, values in boot.items()}
    result = dict(point)
    for metric, (low, high) in ci.items():
        result[f"{metric}_ci_low"] = low
        result[f"{metric}_ci_high"] = high
    return result


def summarize_predictions(
    model_name,
    y_true,
    y_pred,
    y_prob,
    paths,
    num_classes,
    bootstrap_iters,
    seed,
    y_calibrated_prob=None,
    temperature=None,
):
    if y_calibrated_prob is None:
        y_calibrated_prob = y_prob

    image_point = compute_image_metrics(y_true, y_pred, y_prob, y_calibrated_prob, num_classes)
    image_boot = bootstrap_indexed_metrics(
        len(y_true),
        bootstrap_iters,
        seed,
        lambda idx: compute_image_metrics(
            y_true[idx], y_pred[idx], y_prob[idx], y_calibrated_prob[idx], num_classes
        ),
    )
    image_per_class = compute_per_class_metrics(y_true, y_pred, y_prob, num_classes)
    image_per_class_boot = bootstrap_per_class_metrics(y_true, y_pred, y_prob, num_classes, bootstrap_iters, seed)

    patient_true, patient_pred, patient_prob = aggregate_patient_predictions(paths, y_true, y_calibrated_prob)
    patient_point = compute_patient_metrics(patient_true, patient_pred, patient_prob, num_classes)
    patient_boot = bootstrap_overall_metrics(
        patient_true,
        patient_pred,
        patient_prob,
        num_classes,
        bootstrap_iters,
        seed,
        metric_fn=compute_patient_metrics,
    )
    return {
        "model": model_name,
        "image_n": int(len(y_true)),
        "patient_n": int(len(patient_true)),
        "temperature": temperature,
        "image": add_ci_to_metrics(image_point, image_boot),
        "patient": add_ci_to_metrics(patient_point, patient_boot),
        "image_per_class": add_per_class_ci(image_per_class, image_per_class_boot),
    }


def add_per_class_ci(point, boot):
    result = {}
    for cls_idx, metrics in point.items():
        result[cls_idx] = dict(metrics)
        for metric, samples in boot[cls_idx].items():
            low, high = ci_from_samples(samples)
            result[cls_idx][f"{metric}_ci_low"] = low
            result[cls_idx][f"{metric}_ci_high"] = high
    return result


def find_holdout_weights(base_ckpt_dir, model_name):
    base_model_name, _ = split_model_name(model_name)
    candidates = [
        os.path.join(base_ckpt_dir, model_name, f"best_model_{base_model_name}.pth"),
        os.path.join(base_ckpt_dir, model_name, f"best_model_{model_name}.pth"),
        os.path.join(base_ckpt_dir, model_name, f"{model_name}_checkpoints", f"best_checkpoint_{model_name}.pth"),
        os.path.join(base_ckpt_dir, model_name, f"{base_model_name}_checkpoints", f"best_checkpoint_{base_model_name}.pth"),
        os.path.join(base_ckpt_dir, model_name, f"{model_name}_checkpoints", f"best_checkpoint_{model_name}.pth"),
    ]
    return next((path for path in candidates if os.path.exists(path)), None)


def find_fold_weights(base_ckpt_dir, model_name, fold_idx):
    base_model_name, _ = split_model_name(model_name)
    candidates = [
        os.path.join(base_ckpt_dir, model_name, f"fold_{fold_idx}", f"best_model_{base_model_name}_fold{fold_idx}.pth"),
        os.path.join(base_ckpt_dir, model_name, f"fold_{fold_idx}", f"best_model_{model_name}_fold{fold_idx}.pth"),
        os.path.join(
            base_ckpt_dir,
            model_name,
            f"fold_{fold_idx}",
            f"{model_name}_checkpoints",
            f"best_checkpoint_{model_name}.pth",
        ),
        os.path.join(
            base_ckpt_dir,
            model_name,
            f"fold_{fold_idx}",
            f"{base_model_name}_checkpoints",
            f"best_checkpoint_{base_model_name}.pth",
        ),
    ]
    return next((path for path in candidates if os.path.exists(path)), None)


def find_fold_metrics_path(base_ckpt_dir, model_name, fold_idx):
    path = os.path.join(base_ckpt_dir, model_name, f"fold_{fold_idx}", f"metrics_fold{fold_idx}.json")
    return path if os.path.exists(path) else None


def resolve_experiment_img_size(base_ckpt_dir, model_name, fallback_size):
    base_model_name, _ = split_model_name(model_name)
    if base_model_name == "DINOV2":
        if fallback_size != 518:
            print("  [WARN] DINOV2 timm model expects 518x518 input; overriding image_size to 518.")
        return 518

    metrics_path = find_fold_metrics_path(base_ckpt_dir, model_name, 1)
    if metrics_path is None:
        return fallback_size
    try:
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        return int(
            metrics.get("experiment", {})
            .get("training_config", {})
            .get("dinov2_img_size", fallback_size)
        )
    except Exception as exc:
        print(f"  [WARN] could not read dinov2_img_size from {metrics_path}: {exc}")
        return fallback_size


def has_fivefold_weights(base_ckpt_dir, model_name, n_splits):
    return any(find_fold_weights(base_ckpt_dir, model_name, fold_idx) for fold_idx in range(1, n_splits + 1))


def run_holdout_eval(model_name, base_ckpt_dir, val_root, args, device):
    weights_path = find_holdout_weights(base_ckpt_dir, model_name)
    if weights_path is None:
        print(f"[SKIP] {model_name}: no holdout weight found under {base_ckpt_dir}")
        return None

    print(f"\nEvaluating {model_name} on holdout set")
    print(f"  weights: {weights_path}")
    image_size = resolve_experiment_img_size(base_ckpt_dir, model_name, args.dinov2_img_size)
    print(f"  image_size={image_size}")
    dataset = CustomImageFolder(val_root, transform=resolve_transform(model_name, image_size))
    y_true, y_pred, y_prob, _logits, paths = evaluate_model_on_dataset(
        model_name,
        weights_path,
        dataset,
        args.batch_size,
        device,
        args.num_classes,
        args.num_workers,
    )
    return summarize_predictions(
        model_name,
        y_true,
        y_pred,
        y_prob,
        paths,
        args.num_classes,
        args.bootstrap_iters,
        args.seed,
        y_calibrated_prob=y_prob,
        temperature=None,
    )


def run_fivefold_test_eval(model_name, base_ckpt_dir, val_root, args, device):
    if not has_fivefold_weights(base_ckpt_dir, model_name, args.n_splits):
        print(f"[SKIP] {model_name}: no fold weights found under {base_ckpt_dir}")
        return None

    print(f"\nEvaluating {model_name} on test set with {args.n_splits}-fold probability ensemble")
    print(f"  test root: {val_root}")
    image_size = resolve_experiment_img_size(base_ckpt_dir, model_name, args.dinov2_img_size)
    print(f"  image_size={image_size}")
    dataset = CustomImageFolder(val_root, transform=resolve_transform(model_name, image_size))
    all_prob = []
    labels_ref, paths_ref = None, None
    used_folds = []

    for fold_idx in range(1, args.n_splits + 1):
        weights_path = find_fold_weights(base_ckpt_dir, model_name, fold_idx)
        if weights_path is None:
            print(f"  [MISS] fold {fold_idx}: no weight file")
            continue

        print(f"  fold {fold_idx}: n={len(dataset)}, weights={weights_path}")
        y_true, _, y_prob, _logits, paths = evaluate_model_on_dataset(
            model_name,
            weights_path,
            dataset,
            args.batch_size,
            device,
            args.num_classes,
            args.num_workers,
        )
        if labels_ref is None:
            labels_ref = y_true
            paths_ref = paths
        elif not np.array_equal(labels_ref, y_true):
            raise RuntimeError("Dataset label order changed while evaluating fold models.")

        all_prob.append(y_prob)
        used_folds.append(fold_idx)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_prob:
        return None

    y_prob = np.mean(np.stack(all_prob, axis=0), axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    result = summarize_predictions(
        f"{model_name}_5fold_test",
        labels_ref,
        y_pred,
        y_prob,
        paths_ref,
        args.num_classes,
        args.bootstrap_iters,
        args.seed,
        y_calibrated_prob=y_prob,
        temperature=None,
    )
    result["folds"] = ",".join(str(fold_idx) for fold_idx in used_folds)
    return result


def metric_cell(result, metric, percent=True):
    value = result[metric]
    low = result[f"{metric}_ci_low"]
    high = result[f"{metric}_ci_high"]
    if percent:
        return f"{value * 100:.1f}% ({low * 100:.1f}%-{high * 100:.1f}%)"
    return f"{value:.3f} ({low:.3f}-{high:.3f})"


def percent_metric_cell(result, metric):
    value = result[metric]
    low = result[f"{metric}_ci_low"]
    high = result[f"{metric}_ci_high"]
    return f"{value * 100:.1f}% ({low * 100:.1f}%-{high * 100:.1f}%)"


def support_cell(result):
    value = result["support"]
    low = result["support_ci_low"]
    high = result["support_ci_high"]
    return f"{value:.0f} ({low:.0f}-{high:.0f})"


def display_output_model_name(model_name):
    normalized = model_name.strip()
    for suffix in ("_5fold_test", "_holdout"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def normalize_output_model_name(model_name):
    normalized = display_output_model_name(model_name)
    return OUTPUT_MODEL_ORDER_ALIASES.get(normalized, normalized)


def sort_results_by_model_order(results):
    def sort_key(index_and_result):
        index, result = index_and_result
        model_name = normalize_output_model_name(result["model"])
        order = OUTPUT_MODEL_ORDER_INDEX.get(model_name)
        if order is None:
            return len(OUTPUT_MODEL_ORDER), index
        return order, index

    return [result for _, result in sorted(enumerate(results), key=sort_key)]


def make_output_tables(results):
    results = sort_results_by_model_order(results)
    image_rows = []
    patient_rows = []
    image_class_rows = []
    raw_rows = []

    for result in results:
        model = display_output_model_name(result["model"])
        image = result["image"]
        patient = result["patient"]

        image_rows.append(
            {
                "Model": model,
                "Image N": result["image_n"],
                "Accuracy (95% CI)": metric_cell(image, "accuracy", percent=True),
                "Sensitivity (95% CI)": metric_cell(image, "sensitivity", percent=True),
                "Specificity (95% CI)": metric_cell(image, "specificity", percent=True),
                "weighted-F1 (95% CI)": metric_cell(image, "weighted_f1", percent=True),
                "macro-F1 (95% CI)": metric_cell(image, "macro_f1", percent=True),
                "Cohen's kappa (95% CI)": metric_cell(image, "kappa", percent=False),
                "macro-AUC (95% CI)": metric_cell(image, "auc", percent=False),
                "ECE (95% CI)": metric_cell(image, "calibrated_ece", percent=False),
                "Brier (95% CI)": metric_cell(image, "calibrated_brier", percent=False),
            }
        )

        patient_rows.append(
            {
                "Model": model,
                "Patient N": result["patient_n"],
                "Accuracy (95% CI)": metric_cell(patient, "accuracy", percent=True),
                "Macro-sensitivity / macro-recall (95% CI)": metric_cell(patient, "sensitivity", percent=True),
                "Macro-specificity (95% CI)": metric_cell(patient, "specificity", percent=True),
                "macro-F1 (95% CI)": metric_cell(patient, "macro_f1", percent=True),
                "Cohen's kappa (95% CI)": metric_cell(patient, "kappa", percent=False),
                "macro-AUC (95% CI)": metric_cell(patient, "macro_auc", percent=False),
            }
        )

        raw_rows.append(flatten_raw_result(result))

    for class_idx, class_name in enumerate(NEW_CLASS_NAMES):
        for result in results:
            model = display_output_model_name(result["model"])
            image_cls = result["image_per_class"][class_idx]
            image_class_rows.append(make_class_row("Image", class_name, model, image_cls))

    all_results = build_all_results_table(image_rows, image_class_rows, patient_rows)

    return {
        "all_results": all_results,
        "image_overall": pd.DataFrame(image_rows),
        "patient_overall": pd.DataFrame(patient_rows),
        "image_per_class": pd.DataFrame(image_class_rows),
        "raw_overall": pd.DataFrame(raw_rows),
    }


def build_all_results_table(image_rows, image_class_rows, patient_rows):
    image_overall_columns = [
        "Section",
        "Model",
        "Image N",
        "Accuracy (95% CI)",
        "Sensitivity (95% CI)",
        "Specificity (95% CI)",
        "weighted-F1 (95% CI)",
        "macro-F1 (95% CI)",
        "Cohen's kappa (95% CI)",
        "macro-AUC (95% CI)",
        "ECE (95% CI)",
        "Brier (95% CI)",
    ]
    image_class_columns = [
        "Section",
        "Level",
        "Class",
        "Model",
        "Prec (95% CI)",
        "Sens (95% CI)",
        "Spec (95% CI)",
        "F1 (95% CI)",
        "Class AUC (95% CI)",
        "Support (95% CI)",
    ]
    patient_overall_columns = [
        "Section",
        "Model",
        "Patient N",
        "Accuracy (95% CI)",
        "Macro-sensitivity / macro-recall (95% CI)",
        "Macro-specificity (95% CI)",
        "macro-F1 (95% CI)",
        "Cohen's kappa (95% CI)",
        "macro-AUC (95% CI)",
    ]

    rows = []
    base_columns = image_overall_columns
    max_len = max(
        len(image_overall_columns),
        len(image_class_columns),
        len(patient_overall_columns),
    )

    def pad(values):
        return values + [""] * (max_len - len(values))

    def append_section(section, columns, section_rows, include_header):
        if include_header:
            rows.append(pad(columns))
        for row in section_rows:
            values = [section if column == "Section" else row.get(column, "") for column in columns]
            rows.append(pad(values))

    append_section("Image-level overall", image_overall_columns, image_rows, include_header=False)
    rows.append([""] * max_len)
    append_section("Image-level per-class metrics", image_class_columns, image_class_rows, include_header=True)
    rows.append([""] * max_len)
    append_section("Patient-level overall", patient_overall_columns, patient_rows, include_header=True)

    return pd.DataFrame(rows, columns=pad(base_columns)).fillna("")


def make_class_row(level, class_name, model, metrics):
    return {
        "Level": level,
        "Class": class_name,
        "Model": model,
        "Prec (95% CI)": percent_metric_cell(metrics, "pre"),
        "Sens (95% CI)": percent_metric_cell(metrics, "sens"),
        "Spec (95% CI)": percent_metric_cell(metrics, "spec"),
        "F1 (95% CI)": percent_metric_cell(metrics, "f1"),
        "Class AUC (95% CI)": metric_cell(metrics, "auc", percent=False),
        "Support (95% CI)": support_cell(metrics),
    }


def flatten_raw_result(result):
    row = {
        "model": display_output_model_name(result["model"]),
        "image_n": result["image_n"],
        "patient_n": result["patient_n"],
    }
    for prefix in ("image", "patient"):
        for key, value in result[prefix].items():
            if key in ("raw_ece", "raw_brier"):
                continue
            if key == "calibrated_ece":
                key = "ece"
            elif key == "calibrated_brier":
                key = "brier"
            row[f"{prefix}_{key}"] = value
    return row


def metric_console(metrics, metric):
    return (
        f"{metrics[metric] * 100:.1f}% "
        f"(95%CI {metrics[f'{metric}_ci_low'] * 100:.1f}%-{metrics[f'{metric}_ci_high'] * 100:.1f}%)"
    )


def support_console(metrics):
    return (
        f"{metrics['support']:.1f} "
        f"(95%CI {metrics['support_ci_low']:.0f}-{metrics['support_ci_high']:.0f})"
    )


def print_per_class_by_class(results, level_key, title):
    print(f"\n{title}:")
    for class_idx, class_name in enumerate(NEW_CLASS_NAMES):
        print(f"  {class_name}:")
        for result in results:
            metrics = result[level_key][class_idx]
            print(f"    Model={display_output_model_name(result['model'])}:")
            print(
                f"      Prec={metric_console(metrics, 'pre')}, "
                f"Sens={metric_console(metrics, 'sens')}, "
                f"Spec={metric_console(metrics, 'spec')}, "
                f"F1={metric_console(metrics, 'f1')}, "
                f"AUC={metric_cell(metrics, 'auc', percent=False)}, "
                f"Support={support_console(metrics)}"
            )


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = sort_results_by_model_order(results)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tables = make_output_tables(results)
    xlsx_path = os.path.join(output_dir, f"bootstrap_eval_ci_{stamp}.xlsx")
    csv_path = os.path.join(output_dir, f"bootstrap_eval_ci_{stamp}.csv")

    with pd.ExcelWriter(xlsx_path) as writer:
        tables["all_results"].to_excel(writer, sheet_name="all_results", index=False)
        for sheet_name, df in tables.items():
            if sheet_name != "all_results":
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    tables["all_results"].to_csv(csv_path, index=False)

    print("\nImage-level overall:")
    print(tables["image_overall"].to_string(index=False))
    print("\nPatient-level overall:")
    print(tables["patient_overall"].to_string(index=False))
    print_per_class_by_class(results, "image_per_class", "Image-level per-class metrics")
    print(f"\nSaved Excel: {xlsx_path}")
    print(f"Saved CSV:   {csv_path}")


def parse_models(models_arg):
    if len(models_arg) == 1 and "," in models_arg[0]:
        return [item.strip() for item in models_arg[0].split(",") if item.strip()]
    return models_arg


def expand_requested_models(base_ckpt_dir, models):
    expanded = []
    for model_name in models:
        expanded.extend(discover_model_experiments(base_ckpt_dir, model_name))
    deduped = []
    seen = set()
    for model_name in expanded:
        if model_name not in seen:
            deduped.append(model_name)
            seen.add(model_name)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained weights and bootstrap 95% CIs.")
    parser.add_argument(
        "--base-ckpt-dir",
        default=str(PROJECT_ROOT / "outputs" / "checkpoints"),
        help="Checkpoint root. Can be single-holdout or 5-fold layout.",
    )
    parser.add_argument(
        "--val-root",
        default=str(PROJECT_ROOT / "data" / "test"),
        help="Validation/test root for holdout evaluation.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "evaluation"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["DINOV2", "DINOv3_base", "CTransPath", "clip_prompt"],
        help="Model names, separated by spaces or a single comma-separated string.",
    )
    parser.add_argument("--eval-mode", choices=["auto", "holdout", "fivefold_test"], default="auto")
    parser.add_argument(
        "--calibration-mode",
        choices=["metrics", "fit", "none"],
        default="none",
        help=(
            "Deprecated and ignored. Bootstrap evaluation now uses raw softmax/ensemble "
            "probabilities only, so calibrated metrics equal raw metrics."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dinov2-img-size", type=int, default=518, help="Input size for DINOV2/DINOv3_base.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    models_to_eval = expand_requested_models(args.base_ckpt_dir, parse_models(args.models))

    print(f"Device: {device}")
    print(f"Checkpoint root: {args.base_ckpt_dir}")
    print(f"Eval mode: {args.eval_mode}")
    print(f"Models to evaluate: {models_to_eval}")
    print(f"Bootstrap iterations: {args.bootstrap_iters}")
    if args.calibration_mode != "none":
        print("Calibration mode is ignored; bootstrap evaluation uses raw probabilities only.")

    results = []
    for model_name in models_to_eval:
        mode = args.eval_mode
        if mode == "auto":
            mode = "fivefold_test" if has_fivefold_weights(args.base_ckpt_dir, model_name, args.n_splits) else "holdout"

        if mode == "fivefold_test":
            result = run_fivefold_test_eval(model_name, args.base_ckpt_dir, args.val_root, args, device)
        else:
            result = run_holdout_eval(model_name, args.base_ckpt_dir, args.val_root, args, device)

        if result is not None:
            results.append(result)

    if not results:
        raise SystemExit("No models were evaluated. Check --base-ckpt-dir, --models, and --eval-mode.")

    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
