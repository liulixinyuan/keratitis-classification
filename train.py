import sys
import os
import argparse

"""
Portable training entry point for the public repository.

Default local layout:
  data/train_transformed/<class_name>/*
  data/test/<class_name>/*
  weights/dinov2/model.safetensors
  weights/dinov3/{config.json, model.safetensors, ...}
  weights/ctranspath/ctranspath.pth
  weights/clip-vit-base-patch32/{config.json, model.safetensors, ...}

All data, output, and foundation-model weight locations can be overridden
through command-line arguments. Patient identifiers are inferred from the
filename prefix before the first underscore, matching the original pipeline.
"""

REPOSITORY_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)
import os
import copy
import json
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, Dataset
# from tqdm import tqdm
from torchvision import models, transforms, datasets
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             cohen_kappa_score, roc_auc_score, confusion_matrix,
                             classification_report, brier_score_loss)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold, StratifiedKFold
from torchvision.models import (ResNet50_Weights, VGG16_Weights, ResNet18_Weights,
                                ResNet34_Weights, Inception_V3_Weights, DenseNet121_Weights,
                                ResNet101_Weights, ViT_B_16_Weights, EfficientNet_B0_Weights,
                                EfficientNet_B4_Weights, EfficientNet_B7_Weights,
                                EfficientNet_V2_L_Weights, EfficientNet_V2_S_Weights)
import time
from collections import Counter, defaultdict
import timm
import gc
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPModel, CLIPTokenizer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from checkpoint_manager import CheckpointManager
from PIL import Image

"""
2026.5.6 改动
1. 基于train_transformed实现五折交叉验证
2. 患者级别隔离：同一患者所有图像必须在同一折
3. 验证集禁用增强：验证集只使用原图，训练集使用原图+增强图
4. 新增各类别precision/recall/F1
5. 新增校准指标：ECE、Brier Score
6. 新增Patient-level evaluation
7. 早停机制：连续10个epoch AUC不提升则停止
2026.5.7改动
1.标签平滑、权重衰减
2026.5.8改动
1.恢复训练机制
2.多卡训练
"""

# ==================== 类别映射 ====================
CLASS_MAPPING = {
    'NC': 0,
    'BK': 1,
    'FK': 2,
    'VK': 3,
    'NIFK': 4,
    'AMK': 5
}

NEW_CLASS_NAMES = ['NC', 'BK', 'FK', 'VK', 'NIFK', 'AMK']
NUM_CLASSES = 6


# ==================== GitHub-friendly path configuration ====================
PROJECT_ROOT = REPOSITORY_ROOT
TRAIN_TRANSFORMED_PATH = os.path.join(PROJECT_ROOT, "data", "train_transformed")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "test")
SAVE_PATH_BASE = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
DINOV2_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "weights", "dinov2", "model.safetensors")
DINOV3_WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights", "dinov3")
CTRANSPATH_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "weights", "ctranspath", "ctranspath.pth")
CLIP_LOCAL_PATH = os.path.join(PROJECT_ROOT, "weights", "clip-vit-base-patch32")
DEFAULT_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "model_configs.json",
)
PUBLIC_MODEL_CONFIG = {}


# ==================== 自定义Dataset ====================
class PatientImageDataset(Dataset):
    """基于文件路径列表的Dataset，支持传入transform"""
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ==================== 患者级数据构建 ====================
def build_patient_data(root_dir):
    """
    从train_transformed目录构建患者级数据结构
    返回: dict {patient_id: {'label': int, 'original': [paths], 'augmented': [paths]}}
    """
    patient_data = {}
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = CLASS_MAPPING.get(class_name)
        if label is None:
            continue

        for filename in os.listdir(class_dir):
            # 跳过非图片文件
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            patient_id = filename.split('_')[0]
            filepath = os.path.join(class_dir, filename)

            if patient_id not in patient_data:
                patient_data[patient_id] = {'label': label, 'original': [], 'augmented': []}

            # 判断是否为增强图：包含 _aug_ 的就是增强图
            if '_aug_' in filename:
                patient_data[patient_id]['augmented'].append(filepath)
            else:
                patient_data[patient_id]['original'].append(filepath)

    return patient_data


def patient_stratified_kfold(patient_data, n_splits=5, random_state=42):
    """
    患者级五折交叉验证：
    1. 同一患者的图像只能出现在同一个fold中；
    2. 按患者的原图数量做贪心分配，使每个类别在各fold中的原图数尽量接近。

    这里按原图数量均衡，而不是按患者数量均衡。因为验证集只使用原图，
    这样能让每一折验证集的图像数量和类别图像分布更接近 1/n_splits。
    返回: 生成器，每次yield (train_patient_ids, val_patient_ids)
    """
    rng = np.random.default_rng(random_state)
    all_patient_ids = list(patient_data.keys())

    patients_by_class = defaultdict(list)
    for pid, info in patient_data.items():
        original_count = len(info['original'])
        if original_count == 0:
            raise ValueError(f"患者 {pid} 没有原图，无法参与按原图数量均衡的五折划分")
        patients_by_class[info['label']].append((pid, original_count))

    fold_patient_ids = [[] for _ in range(n_splits)]
    fold_total_original_counts = [0 for _ in range(n_splits)]
    fold_class_original_counts = [
        [0 for _ in range(NUM_CLASSES)] for _ in range(n_splits)
    ]

    for label in range(NUM_CLASSES):
        class_patients = patients_by_class[label]
        if len(class_patients) < n_splits:
            raise ValueError(
                f"类别 {NEW_CLASS_NAMES[label]} 的患者数只有 {len(class_patients)}，"
                f"少于折数 {n_splits}，无法保证每折都有该类别患者"
            )

        # 先随机打散同图像数患者，再按原图数从多到少放置，减少大患者造成的失衡。
        shuffled_indices = rng.permutation(len(class_patients))
        class_patients = [class_patients[i] for i in shuffled_indices]
        class_patients.sort(key=lambda item: item[1], reverse=True)

        for pid, original_count in class_patients:
            # 优先补齐当前类别图像数更少的fold；若相同，再补齐总原图数更少的fold。
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
            raise RuntimeError("五折划分患者数量校验失败")
        yield train_pids, val_pids


def build_fold_datasets(patient_data, train_pids, val_pids, train_transform, val_transform):
    """
    构建训练集和验证集Dataset
    训练集：使用患者的所有图像（原图+增强图）
    验证集：只使用患者的原图（不含增强图）
    """
    # 训练集：所有图像
    train_paths = []
    train_labels = []
    for pid in train_pids:
        all_paths = patient_data[pid]['original'] + patient_data[pid]['augmented']
        train_paths.extend(all_paths)
        train_labels.extend([patient_data[pid]['label']] * len(all_paths))

    # 验证集：只使用原图
    val_paths = []
    val_labels = []
    val_patient_ids_per_sample = []
    for pid in val_pids:
        orig_paths = patient_data[pid]['original']
        val_paths.extend(orig_paths)
        val_labels.extend([patient_data[pid]['label']] * len(orig_paths))
        val_patient_ids_per_sample.extend([pid] * len(orig_paths))

    train_dataset = PatientImageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PatientImageDataset(val_paths, val_labels, transform=val_transform)

    return train_dataset, val_dataset, val_patient_ids_per_sample


# ==================== 评估工具函数 ====================
def compute_ece(y_true, y_prob, n_bins=15):
    """
    计算Expected Calibration Error (ECE)
    y_true: [N] int labels
    y_prob: [N, C] softmax probabilities
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def compute_brier_score(y_true, y_prob, n_classes):
    """
    计算多分类Brier Score
    """
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    brier = 0.0
    for c in range(n_classes):
        brier += brier_score_loss(y_true_bin[:, c], y_prob[:, c])
    return brier / n_classes


def fit_temperature_scaling(logits, labels, device):
    """
    在验证集logits上拟合单个温度参数，返回校准后的概率和温度值。
    """
    logits_tensor = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    log_temperature = torch.nn.Parameter(torch.zeros(1, device=device))
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=100.0)
        loss = nll_criterion(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = torch.exp(log_temperature).clamp(min=1e-3, max=100.0).detach()
    calibrated_probs = torch.softmax(logits_tensor / temperature, dim=1).detach().cpu().numpy()
    return calibrated_probs, float(temperature.item())


def patient_level_evaluation(val_patient_ids, all_labels, all_probs):
    """
    患者级评估：对同一患者的所有图像softmax概率逐类平均
    返回: (patient_accuracy, patient_cm, patient_true_labels, patient_pred_labels, patient_avg_probs)
    """
    patient_probs = defaultdict(lambda: {'probs': [], 'label': None})

    for pid, label, prob in zip(val_patient_ids, all_labels, all_probs):
        patient_probs[pid]['probs'].append(prob)
        if patient_probs[pid]['label'] is None:
            patient_probs[pid]['label'] = label

    patient_pred_labels = []
    patient_true_labels = []
    patient_avg_probs_list = []

    for pid in sorted(patient_probs.keys()):
        data = patient_probs[pid]
        avg_prob = np.mean(data['probs'], axis=0)
        pred_label = int(np.argmax(avg_prob))
        patient_pred_labels.append(pred_label)
        patient_true_labels.append(data['label'])
        patient_avg_probs_list.append(avg_prob)

    accuracy = accuracy_score(patient_true_labels, patient_pred_labels)
    cm = confusion_matrix(patient_true_labels, patient_pred_labels)
    return accuracy, cm, patient_true_labels, patient_pred_labels, np.array(patient_avg_probs_list)


def compute_sensitivity_specificity(y_true, y_pred, n_classes):
    """
    计算各类别的敏感性(Sensitivity/Recall)和特异性(Specificity)
    返回: (sensitivities, specificities) 每个都是长度为n_classes的列表
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    sensitivities = []
    specificities = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    return sensitivities, specificities


def compute_per_class_metrics(y_true, y_pred, class_names):
    """计算各类别的precision, recall, f1, sensitivity, specificity"""
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    n_classes = len(class_names)
    sensitivities, specificities = compute_sensitivity_specificity(y_true, y_pred, n_classes)
    
    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            'precision': report[name]['precision'],
            'recall': report[name]['recall'],
            'sensitivity': sensitivities[idx],
            'specificity': specificities[idx],
            'f1-score': report[name]['f1-score'],
            'support': report[name]['support']
        }
    return per_class


# ==================== 可序列化的 ProcessorTransform（用于分布式） ====================
class ProcessorTransform:
    def __init__(self, local_dir=None, fallback_id="facebook/dinov2-base"):
        self.local_dir = local_dir
        self.fallback_id = fallback_id
        self._processor = None

    def _ensure_loaded(self):
        if self._processor is not None:
            return
        try:
            if self.local_dir is not None:
                preprocessor_config_path = os.path.join(self.local_dir, "preprocessor_config.json")
                config_json_path = os.path.join(self.local_dir, "config.json")
                if not os.path.exists(preprocessor_config_path) and not os.path.exists(config_json_path):
                    raise FileNotFoundError(
                        f"在 {self.local_dir} 中未找到 preprocessor_config.json 或 config.json 文件。"
                    )
                self._processor = AutoImageProcessor.from_pretrained(self.local_dir, local_files_only=True)
            else:
                raise ValueError("No local_dir provided for ProcessorTransform, and online loading is disabled.")
        except Exception as e_local:
            print(f"⚠️ AutoImageProcessor 从本地目录加载失败: {self.local_dir} - {e_local}")
            raise RuntimeError("无法加载图像处理器：本地文件不存在或格式不正确，且已禁用在线下载。") from e_local

    def __call__(self, pil_img):
        self._ensure_loaded()
        data = self._processor(images=pil_img, return_tensors="pt")
        return data["pixel_values"].squeeze(0)


class SmartToTensor:
    def __call__(self, pic):
        if not isinstance(pic, torch.Tensor):
            return transforms.ToTensor()(pic)
        return pic


class CustomClassifier(nn.Module):
    def __init__(self, base_model, in_dim, num_classes):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes)
        )

    def forward(self, x):
        x = self.base(x)
        return self.classifier(x)


class CLIPPromptClassifier(nn.Module):
    def __init__(self, class_names, clip_path=None, prompt_templates=None, protocol="zero_shot"):
        super().__init__()
        assert clip_path is not None, "必须传入 clip_path"
        self.protocol = protocol
        self.clip = CLIPModel.from_pretrained(clip_path, local_files_only=True, use_safetensors=False)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
        self.prompt_templates = prompt_templates or DEFAULT_CLIP_PROMPT_TEMPLATES
        proj_dim = self.clip.config.projection_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, len(class_names))
        )
        prompts = build_clip_prompts(class_names, self.prompt_templates)
        text_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        with torch.no_grad():
            text_features = self.clip.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(len(class_names), len(self.prompt_templates), -1).mean(dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.register_buffer("text_features", text_features, persistent=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(100.0))

    def forward(self, x):
        image_features = self.clip.get_image_features(pixel_values=x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if self.protocol == "zero_shot":
            logits = self.logit_scale.exp() * image_features @ self.text_features.T
        else:
            logits = self.classifier(image_features)
        return logits


class PortableDINOv3Model(nn.Module):
    """DINOv3 classifier that loads its backbone from a configurable local directory."""
    def __init__(self, num_classes, adaptation="full_finetune", partial_blocks=2):
        super().__init__()
        if not os.path.isdir(DINOV3_WEIGHT_DIR):
            raise FileNotFoundError(
                f"DINOv3 weight directory not found: {DINOV3_WEIGHT_DIR}. "
                "Pass --dinov3-weight-dir."
            )
        self.backbone = AutoModel.from_pretrained(DINOV3_WEIGHT_DIR, local_files_only=True)
        embed_dim = getattr(self.backbone.config, "hidden_size", 768)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )
        self.apply_adaptation_strategy(adaptation, partial_blocks)

    def apply_adaptation_strategy(self, adaptation, partial_blocks=2):
        if adaptation == "full_finetune":
            set_requires_grad(self, True)
        elif adaptation == "linear_probe":
            set_requires_grad(self.backbone, False)
            set_requires_grad(self.classifier, True)
        elif adaptation == "partial_finetune":
            set_requires_grad(self.backbone, False)
            set_requires_grad(self.classifier, True)
            unfreeze_last_blocks(self.backbone, partial_blocks)
        else:
            raise ValueError(f"DINOv3 does not support adaptation={adaptation}")

    def forward(self, x):
        features = self.backbone(x)
        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state[:, 0]
        elif isinstance(features, tuple):
            features = features[0]
        if features.dim() == 4:
            features = features.mean(dim=(2, 3))
        elif features.dim() == 3:
            features = features[:, 0]
        return self.classifier(features)


DEFAULT_CLIP_PROMPT_TEMPLATES = [
    "a slit-lamp photograph of {}.",
    "a clinical slit-lamp image showing {}.",
    "an anterior segment slit-lamp photograph diagnosed as {}.",
    "a corneal disease slit-lamp image of {}.",
    "a close-up slit-lamp photograph of the cornea showing {}.",
]


CLIP_CLASS_TEXT = {
    "NC": "a normal cornea",
    "BK": "bacterial keratitis",
    "FK": "fungal keratitis",
    "VK": "viral keratitis",
    "NIFK": "non-infectious keratitis",
    "AMK": "Acanthamoeba keratitis",
}


def build_clip_prompts(class_names, prompt_templates):
    return [
        template.format(CLIP_CLASS_TEXT.get(class_name, class_name))
        for class_name in class_names
        for template in prompt_templates
    ]


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def unfreeze_last_blocks(module, block_count):
    blocks = getattr(module, "blocks", None)
    if blocks is None and hasattr(module, "encoder"):
        blocks = getattr(module.encoder, "layer", None)
    if blocks is None and hasattr(module, "encoder"):
        blocks = getattr(module.encoder, "layers", None)
    if blocks is None:
        return
    for block in list(blocks)[-block_count:]:
        set_requires_grad(block, True)


def apply_adaptation_strategy(model, model_name, protocol, partial_blocks=2):
    """
    统一记录并应用基础模型适配策略：
    zero_shot: 不训练，仅限 CLIP prompt；
    linear_probe: 冻结 backbone，仅训练分类头；
    partial_finetune: 冻结大部分 backbone，仅解冻末端 blocks + 分类头；
    full_finetune / few_shot: 全量微调，few_shot 只改变训练数据量。
    """
    if hasattr(model, "module"):
        model = model.module

    if model_name == "clip_prompt":
        if protocol == "zero_shot":
            set_requires_grad(model, False)
        elif protocol == "linear_probe":
            set_requires_grad(model.clip, False)
            set_requires_grad(model.classifier, True)
            model.logit_scale.requires_grad = False
        elif protocol == "partial_finetune":
            set_requires_grad(model.clip, False)
            vision_model = getattr(model.clip, "vision_model", None)
            if vision_model is not None:
                unfreeze_last_blocks(vision_model, partial_blocks)
                if hasattr(vision_model, "post_layernorm"):
                    set_requires_grad(vision_model.post_layernorm, True)
            model.logit_scale.requires_grad = True
        elif protocol in ["full_finetune", "few_shot"]:
            set_requires_grad(model.clip, True)
            set_requires_grad(model.classifier, True)
            model.logit_scale.requires_grad = True
        else:
            raise ValueError(f"clip_prompt 不支持 protocol={protocol}")
    elif model_name == "DINOv3_base":
        # DINOv3ModelParallel 已在构造函数中支持适配策略，这里只做兜底检查。
        pass
    elif model_name in ["DINOV2", "CTransPath"]:
        if protocol == "zero_shot":
            raise ValueError(f"{model_name} 没有文本提示词分类器，zero_shot 不适用。")
        if protocol == "linear_probe":
            if model_name == "CTransPath":
                set_requires_grad(model.base, False)
                set_requires_grad(model.classifier, True)
            else:
                set_requires_grad(model, False)
                set_requires_grad(model.head, True)
        elif protocol == "partial_finetune":
            if model_name == "CTransPath":
                set_requires_grad(model.base, False)
                set_requires_grad(model.classifier, True)
                unfreeze_last_blocks(model.base, partial_blocks)
                if hasattr(model.base, "norm"):
                    set_requires_grad(model.base.norm, True)
            else:
                set_requires_grad(model, False)
                set_requires_grad(model.head, True)
                unfreeze_last_blocks(model, partial_blocks)
                if hasattr(model, "norm"):
                    set_requires_grad(model.norm, True)
        elif protocol in ["full_finetune", "few_shot"]:
            set_requires_grad(model, True)
        else:
            raise ValueError(f"{model_name} 不支持 protocol={protocol}")
    else:
        if protocol in ["zero_shot", "linear_probe", "partial_finetune", "few_shot"]:
            raise ValueError(f"{model_name} 不是本返修脚本的基础模型，protocol={protocol} 不适用。")
        set_requires_grad(model, True)


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_experiment_name(model_name, protocol, shots):
    if protocol == "few_shot":
        return f"{model_name}_{protocol}_{shots}shot"
    return f"{model_name}_{protocol}"


def build_few_shot_train_dataset(train_dataset, shots_per_class, seed=42):
    """
    从每个训练 fold 中按类别抽取 k 张原图；若某类原图不足，再用增强图补齐。
    """
    rng = np.random.default_rng(seed)
    selected_paths, selected_labels = [], []
    for label in range(NUM_CLASSES):
        label_items = [
            (path, y) for path, y in zip(train_dataset.file_paths, train_dataset.labels)
            if y == label
        ]
        original_items = [item for item in label_items if "_aug_" not in os.path.basename(item[0])]
        augmented_items = [item for item in label_items if "_aug_" in os.path.basename(item[0])]

        rng.shuffle(original_items)
        rng.shuffle(augmented_items)
        chosen = original_items[:shots_per_class]
        if len(chosen) < shots_per_class:
            chosen.extend(augmented_items[:shots_per_class - len(chosen)])
        if len(chosen) == 0:
            raise ValueError(f"few-shot 抽样失败：类别 {NEW_CLASS_NAMES[label]} 在当前训练 fold 中无样本")

        selected_paths.extend([path for path, _ in chosen])
        selected_labels.extend([y for _, y in chosen])

    return PatientImageDataset(selected_paths, selected_labels, transform=train_dataset.transform)


# ==================== 模型选择和初始化 ====================
def get_model(model_name, num_classes, class_names=None, protocol="full_finetune",
              partial_blocks=2, prompt_templates=None, device_ids=[4, 5, 6, 7]):
    if model_name == "AlexNet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif model_name == "VGG16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif model_name == "InceptionV3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False
        # 彻底删除 AuxLogits 分支，避免 DataParallel 多卡时 batch_size=1 导致 BatchNorm 报错
        model.AuxLogits = None
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif model_name == "ResNeXt50":
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "RegNet":
        model = models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "EfficientNet_B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name == "ConvNeXT":
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)
    elif model_name == "ViT":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
    elif model_name == "DINOV2":
        model = timm.create_model("vit_small_patch14_dinov2",
                                  pretrained=False,
                                  checkpoint_path=DINOV2_CHECKPOINT_PATH)
        num_features = model.num_features
        model.head = nn.Linear(num_features, num_classes)
    elif model_name == "DINOv3_base":
        dino_adaptation = "full_finetune" if protocol == "few_shot" else protocol
        model = PortableDINOv3Model(
            num_classes,
            adaptation=dino_adaptation,
            partial_blocks=partial_blocks,
        )
        return model
    elif model_name == "clip_prompt":
        assert class_names is not None, "clip_prompt 模式需要传入 class_names"
        model = CLIPPromptClassifier(class_names=class_names,
                                     clip_path=CLIP_LOCAL_PATH,
                                     prompt_templates=prompt_templates,
                                     protocol=protocol)
    elif model_name == "CTransPath":
        model = timm.create_model(
            'vit_small_patch16_224',
            img_size=224, patch_size=16, embed_dim=384,
            depth=12, num_heads=6, num_classes=0, pretrained=False
        )
        possible_paths = [CTRANSPATH_CHECKPOINT_PATH]
        ctranspath_loaded = False
        for ctranspath_path in possible_paths:
            if os.path.exists(ctranspath_path):
                try:
                    state_dict = torch.load(ctranspath_path, map_location="cpu")
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    model_keys = set(model.state_dict().keys())
                    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
                    model.load_state_dict(state_dict, strict=False)
                    print(f"CTransPath预训练权重加载成功: {ctranspath_path}")
                    ctranspath_loaded = True
                    break
                except Exception as e:
                    print(f"加载CTransPath权重失败 {ctranspath_path}: {e}")
                    continue
        if not ctranspath_loaded:
            print("警告: 未找到CTransPath预训练权重文件，将使用随机初始化的权重")
        custom_model = CustomClassifier(model, in_dim=384, num_classes=num_classes)
        model = custom_model
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    apply_adaptation_strategy(model, model_name, protocol, partial_blocks=partial_blocks)
    return model


# ==================== 带早停的训练函数 ====================
def train_and_evaluate(model, train_loader, val_loader, num_epochs, criterion, optimizer, device,
                       save_path=None, model_name=None, resume_from_checkpoint=None, scheduler=None,
                       early_stop_patience=24, scheduler_name="plateau", use_amp=True,
                       grad_accum_steps=1, rank=0, distributed=False):
    """
    训练并评估模型，支持早停机制
    early_stop_patience: 连续多少个epoch AUC不提升则早停
    返回: (trained_model, best_auc, train_history, stopped_early)
    """
    checkpoint_manager = None
    main_process = rank == 0
    if save_path and model_name:
        checkpoint_manager = CheckpointManager(save_path, model_name)

    start_epoch = 0
    best_auc = float('-inf')
    best_model_weights = None
    epochs_no_improve = 0
    stopped_early = False

    train_history = {
        'train_loss': [], 'val_loss': [],
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'kappa': [], 'auc': [],
        'epoch_time': []
    }

    # 尝试恢复 checkpoint
    if resume_from_checkpoint and checkpoint_manager:
        try:
            checkpoint, start_epoch, best_auc = checkpoint_manager.load_checkpoint(
                resume_from_checkpoint, device
            )
            # 处理 DataParallel 的 state_dict 键名
            state_dict = checkpoint['model_state_dict']
            if hasattr(model, 'module'):
                # 如果当前是 DataParallel，但 checkpoint 带 module. 前缀，需要去掉
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
                    else:
                        new_state_dict[k] = v
                model.module.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'train_history' in checkpoint:
                train_history = checkpoint['train_history']
            if main_process:
                print(f"✅ 成功恢复训练，从第 {start_epoch + 1} 个 epoch 开始 | 当前最佳 Val AUC: {best_auc:.4f}")
        except Exception as e:
            if main_process:
                print(f"⚠️ 无法恢复 checkpoint（{e}），将从头开始训练")

    use_amp = bool(use_amp and device.type == "cuda")
    grad_accum_steps = max(1, int(grad_accum_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # ---------- 训练阶段 ----------
        model.train()
        running_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_train_loss += loss.item()
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        avg_train_loss = running_train_loss / len(train_loader)
        train_history['train_loss'].append(avg_train_loss)

        # ---------- 验证阶段 ----------
        model.eval()
        running_val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                probs = torch.softmax(outputs.float(), dim=1)
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = normalize_probabilities(all_probs)

        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(all_labels, all_preds)

        train_history['val_loss'].append(avg_val_loss)
        train_history['accuracy'].append(accuracy)
        train_history['precision'].append(precision)
        train_history['recall'].append(recall)
        train_history['f1'].append(f1)
        train_history['kappa'].append(kappa)
        train_history['auc'].append(auc)

        # ---------- 保存最优模型 ----------
        is_best = auc > best_auc
        if is_best:
            best_auc = auc
            # DataParallel 的 state_dict 带有 module. 前缀，保存时去掉
            if hasattr(model, 'module'):
                best_model_weights = {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}
            else:
                best_model_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if checkpoint_manager and main_process:
            checkpoint_manager.save_checkpoint(
                epoch=epoch, model=model, optimizer=optimizer,
                scheduler=scheduler, best_metric=best_auc,
                train_history=train_history,
                config={'model_name': model_name, 'num_epochs': num_epochs,
                        'batch_size': train_loader.batch_size,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'device': str(device)},
                is_best=is_best
            )

        epoch_end = time.time()
        train_history['epoch_time'].append(epoch_end - epoch_start)

        # 获取当前学习率（用于打印）
        current_lr = optimizer.param_groups[0]['lr']

        if main_process:
            print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | AUC: {auc:.4f} | Best AUC: {best_auc:.4f} | "
                  f"Acc: {accuracy:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f} | "
                  f"NoImprove: {epochs_no_improve}/{early_stop_patience} | LR: {current_lr:.6f} | Time: {epoch_end - epoch_start:.2f}s")

        if scheduler:
            prev_lr = optimizer.param_groups[0]['lr']
            if scheduler_name == "plateau":
                scheduler.step(auc)
            else:
                scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < prev_lr and main_process:
                print(f"📉 LR 下降: {prev_lr:.6f} → {new_lr:.6f}")

        del all_preds, all_labels, all_probs, inputs, labels, outputs, preds, probs, loss

        # ---------- 早停检查 ----------
        if epochs_no_improve >= early_stop_patience:
            if main_process:
                print(f"⏹️  早停触发！连续 {early_stop_patience} 个 epoch AUC 未提升。")
            stopped_early = True
            break

    # 加载最佳模型
    if best_model_weights is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_model_weights)
        else:
            model.load_state_dict(best_model_weights)
    model.to(device)
    return model, best_auc, train_history, stopped_early


# ==================== 图像变换 ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform2 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform4 = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=clip_mean, std=clip_std)
])

cnn_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inception_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(299, scale=(0.75, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vit_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dinov2_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(518, scale=(0.85, 1.0), ratio=(0.95, 1.05), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.08, contrast=0.08),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def build_dinov2_transforms(image_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

clip_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.08, contrast=0.08),
    transforms.ToTensor(),
    transforms.Normalize(mean=clip_mean, std=clip_std)
])


# ==================== 设备设置 ====================
# 多卡训练支持：
#   单卡: CUDA_VISIBLE_DEVICES=0 python classification/train_asoct_6classes.py
#   多卡: CUDA_VISIBLE_DEVICES=0,1,2,3 python classification/train_asoct_6classes.py
# 注意：device 和 n_gpus 在这里定义，但 GPU 信息打印移到 main() 中避免 import 时重复输出
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    n_gpus = 0


def init_distributed_mode(args):
    """Initialize torchrun/DDP state when launched with torchrun."""
    args.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not args.distributed:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA, but CUDA is not available.")

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    return torch.device(f"cuda:{args.local_rank}")


def cleanup_distributed_mode(args):
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(args=None):
    if args is not None:
        return getattr(args, "rank", 0) == 0
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def main_print(*values, args=None, **kwargs):
    if is_main_process(args):
        print(*values, **kwargs)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def normalize_probabilities(probabilities, eps=1e-12):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    return probabilities / np.clip(row_sums, eps, None)


# 统一随机种子
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ==================== 模型配置 ====================
FOUNDATION_MODEL_NAMES = ["DINOV2", "DINOv3_base", "CTransPath", "clip_prompt"]
BASELINE_MODEL_NAMES = [
    "InceptionV3", "ResNet50", "ResNet18", "MobileNetV2", "DenseNet121",
    "ResNeXt50", "RegNet", "EfficientNet_B0", "ConvNeXT", "ViT"
]
ALL_MODEL_NAMES = FOUNDATION_MODEL_NAMES + BASELINE_MODEL_NAMES
DEFAULT_MODEL_NAMES = FOUNDATION_MODEL_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Foundation-model protocols for slit-lamp keratitis 6-class CV.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Public JSON model-configuration file.")
    parser.add_argument("--train-dir", default=TRAIN_TRANSFORMED_PATH, help="Training directory containing class subdirectories.")
    parser.add_argument("--test-dir", default=TEST_PATH, help="Independent test-set directory.")
    parser.add_argument("--output-dir", default=SAVE_PATH_BASE, help="Directory used to save checkpoints and results.")
    parser.add_argument("--dinov2-checkpoint", default=DINOV2_CHECKPOINT_PATH, help="Path to the DINOv2 model.safetensors file.")
    parser.add_argument("--dinov3-weight-dir", default=DINOV3_WEIGHT_DIR, help="Local Hugging Face-format DINOv3 weight directory.")
    parser.add_argument("--ctranspath-checkpoint", default=CTRANSPATH_CHECKPOINT_PATH, help="Path to the CTransPath pretrained checkpoint.")
    parser.add_argument("--clip-weight-dir", default=CLIP_LOCAL_PATH, help="Local Hugging Face-format CLIP weight directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "要训练的模型名，可传一个或多个，例如: "
            "--models DINOV2 DINOv3_base CTransPath clip_prompt"
        ),
    )
    parser.add_argument(
        "--protocol",
        default=None,
        choices=["zero_shot", "few_shot", "linear_probe", "partial_finetune", "full_finetune"],
        help="只运行一个基础模型评估协议；不传时按模型自动运行返修要求的协议。",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=None,
        choices=["zero_shot", "few_shot", "linear_probe", "partial_finetune", "full_finetune"],
        help="运行多个协议；例如 --protocols linear_probe full_finetune。",
    )
    parser.add_argument("--shots", type=int, default=5, help="few_shot 协议下每类训练图像数。")
    parser.add_argument("--partial-blocks", type=int, default=2, help="partial_finetune 解冻的末端 transformer blocks 数。")
    parser.add_argument("--epochs", type=int, default=50, help="训练 epoch 数；zero_shot 会自动跳过训练。")
    parser.add_argument("--batch-size", type=int, default=128, help="DataLoader batch size。")
    parser.add_argument("--dinov2-img-size", type=int, default=518, help="DINOv2/DINOv3 输入尺寸。")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="梯度累积步数，用小 batch 模拟更大的有效 batch。")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA AMP 混合精度。")
    return parser.parse_args()


def get_default_protocols(model_name):
    if model_name == "clip_prompt":
        return ["zero_shot", "linear_probe", "full_finetune"]
    if model_name in ["DINOV2", "DINOv3_base", "CTransPath"]:
        return ["linear_probe", "full_finetune"]
    return ["full_finetune"]


def build_experiment_jobs(model_names, args):
    jobs = []
    for model_name in model_names:
        if args.protocols is not None:
            protocols = args.protocols
        elif args.protocol is not None:
            protocols = [args.protocol]
        else:
            protocols = get_default_protocols(model_name)

        for protocol in protocols:
            if protocol == "zero_shot" and model_name != "clip_prompt":
                raise ValueError("zero_shot 仅适用于 clip_prompt；DINOv2/DINOv3/CTransPath 无文本提示词分类器。")
            if protocol == "partial_finetune" and model_name not in FOUNDATION_MODEL_NAMES:
                raise ValueError(f"{model_name} 不是基础模型，不支持 partial_finetune。")
            if protocol in ["linear_probe", "few_shot"] and model_name not in FOUNDATION_MODEL_NAMES:
                raise ValueError(f"{model_name} 不是基础模型，不支持 {protocol}。")
            jobs.append((model_name, protocol))
    return jobs


def get_training_config(model_name, protocol):
    config = {
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "scheduler": "plateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 8,
        "augmentation": "cnn_moderate",
        "optimizer": "AdamW",
    }

    if model_name == "ViT":
        config.update({
            "augmentation": "vit_light",
        })
    elif model_name == "DINOV2":
        config.update({
            "augmentation": "dinov2_light",
        })
    elif model_name == "DINOv3_base":
        config.update({
            "augmentation": "dinov2_light",
        })
    elif model_name == "CTransPath":
        config.update({
            "augmentation": "vit_light",
        })
    elif model_name == "clip_prompt":
        config.update({
            "augmentation": "clip_light",
        })
        if protocol == "zero_shot":
            config.update({
                "learning_rate": 0.0,
                "weight_decay": 0.0,
                "scheduler": "none",
                "augmentation": "clip_eval_only",
            })

    model_config = PUBLIC_MODEL_CONFIG.get("models", {}).get(model_name, {})
    protocol_config = model_config.get("protocols", {}).get(protocol)
    if protocol_config is not None:
        config.update(protocol_config)
    return config


def load_public_model_config(config_path):
    if not config_path:
        return {}
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config.get("models"), dict):
        raise ValueError(f"Invalid model configuration file: {config_path}")
    return config


def build_scheduler(optimizer, training_config, epochs):
    scheduler_name = training_config["scheduler"]
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=training_config["learning_rate"] * 0.01
        )
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max',
            factor=training_config["scheduler_factor"],
            patience=training_config["scheduler_patience"]
        )
    raise ValueError(f"不支持的 scheduler: {scheduler_name}")


def get_model_transform(model_name, protocol, training_config):
    """根据模型名称返回对应的训练transform和验证transform"""
    if model_name == "InceptionV3":
        return inception_train_transform, transform2
    elif model_name == "clip_prompt":
        if protocol == "zero_shot":
            return clip_transform, clip_transform
        return clip_train_transform, clip_transform
    elif model_name == "DINOV2":
        return build_dinov2_transforms(training_config.get("dinov2_img_size", 518))
    elif model_name == "DINOv3_base":
        return build_dinov2_transforms(training_config.get("dinov2_img_size", 518))
    elif model_name in ["ViT", "CTransPath"]:
        return vit_train_transform, transform
    else:
        return cnn_train_transform, transform


# ==================== 主函数 ====================
def main():
    global device, n_gpus
    global TRAIN_TRANSFORMED_PATH, TEST_PATH, SAVE_PATH_BASE
    global DINOV2_CHECKPOINT_PATH, DINOV3_WEIGHT_DIR, CTRANSPATH_CHECKPOINT_PATH, CLIP_LOCAL_PATH
    global PUBLIC_MODEL_CONFIG
    args = parse_args()
    PUBLIC_MODEL_CONFIG = load_public_model_config(args.config)
    TRAIN_TRANSFORMED_PATH = os.path.abspath(args.train_dir)
    TEST_PATH = os.path.abspath(args.test_dir)
    SAVE_PATH_BASE = os.path.abspath(args.output_dir)
    DINOV2_CHECKPOINT_PATH = os.path.abspath(args.dinov2_checkpoint)
    DINOV3_WEIGHT_DIR = os.path.abspath(args.dinov3_weight_dir)
    CTRANSPATH_CHECKPOINT_PATH = os.path.abspath(args.ctranspath_checkpoint)
    CLIP_LOCAL_PATH = os.path.abspath(args.clip_weight_dir)
    set_seed(args.seed)

    device = init_distributed_mode(args)
    n_gpus = args.world_size if getattr(args, "distributed", False) else (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    model_names = args.models if args.models else DEFAULT_MODEL_NAMES
    invalid_models = [name for name in model_names if name not in ALL_MODEL_NAMES]
    if invalid_models:
        raise ValueError(
            f"不支持的模型名: {invalid_models}。可选模型: {ALL_MODEL_NAMES}"
        )
    if not os.path.isdir(TRAIN_TRANSFORMED_PATH):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_TRANSFORMED_PATH}")
    if not os.path.isdir(TEST_PATH):
        raise FileNotFoundError(f"Test directory not found: {TEST_PATH}")
    required_weights = {
        "DINOV2": DINOV2_CHECKPOINT_PATH,
        "DINOv3_base": DINOV3_WEIGHT_DIR,
        "CTransPath": CTRANSPATH_CHECKPOINT_PATH,
        "clip_prompt": CLIP_LOCAL_PATH,
    }
    for model_name in model_names:
        path = required_weights.get(model_name)
        if path is not None and not os.path.exists(path):
            raise FileNotFoundError(
                f"Required pretrained weights for {model_name} were not found: {path}"
            )
    if args.protocol == "few_shot" and args.shots <= 0:
        raise ValueError("--shots 必须为正整数")
    if args.protocols is not None and "few_shot" in args.protocols and args.shots <= 0:
        raise ValueError("--shots 必须为正整数")
    experiment_jobs = build_experiment_jobs(model_names, args)

    # 打印 GPU 信息（放在 main 里避免 import 时重复输出）
    if torch.cuda.is_available():
        if getattr(args, "distributed", False):
            main_print(
                f"Using DDP with {args.world_size} process(es): "
                f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}",
                args=args,
            )
        else:
            print(f"Using {n_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")
    else:
        print("Using CPU")

    main_print(f"Save path base: {SAVE_PATH_BASE}", args=args)
    main_print(f"Model configuration: {os.path.abspath(args.config)}", args=args)
    main_print(f"Train transformed path: {TRAIN_TRANSFORMED_PATH}", args=args)
    main_print(f"Test path: {TEST_PATH}", args=args)
    main_print(f"Models to train: {model_names}", args=args)
    main_print(f"Experiment jobs: {experiment_jobs}", args=args)
    if any(protocol == "few_shot" for _, protocol in experiment_jobs):
        main_print(f"Few-shot setting: {args.shots} images/class/fold", args=args)
    if any(protocol == "partial_finetune" for _, protocol in experiment_jobs):
        main_print(f"Partial fine-tuning: unfreeze last {args.partial_blocks} transformer blocks", args=args)
    if "clip_prompt" in model_names:
        main_print("CLIP prompt templates:", args=args)
        for template in DEFAULT_CLIP_PROMPT_TEMPLATES:
            main_print(f"   - {template}", args=args)

    # 构建患者级数据
    main_print("\n📊 正在构建患者级数据...", args=args)
    patient_data = build_patient_data(TRAIN_TRANSFORMED_PATH)
    total_patients = len(patient_data)
    total_original = sum(len(v['original']) for v in patient_data.values())
    total_augmented = sum(len(v['augmented']) for v in patient_data.values())
    main_print(f"   总患者数: {total_patients}", args=args)
    main_print(f"   原图总数: {total_original}", args=args)
    main_print(f"   增强图总数: {total_augmented}", args=args)
    main_print(f"   总图像数: {total_original + total_augmented}", args=args)

    # 统计每类患者数
    class_patient_counts = Counter([v['label'] for v in patient_data.values()])
    main_print(f"   每类患者数: {dict(class_patient_counts)}", args=args)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer_cls = optim.AdamW

    # 遍历每个模型和实验协议
    for model_name, protocol in experiment_jobs:
        main_print(f"\n{'='*60}", args=args)
        main_print(f"🚀 开始实验: {model_name} | {protocol}", args=args)
        main_print(f"{'='*60}", args=args)

        experiment_name = build_experiment_name(model_name, protocol, args.shots)
        model_save_dir = os.path.join(SAVE_PATH_BASE, experiment_name)
        os.makedirs(model_save_dir, exist_ok=True)
        training_config = get_training_config(model_name, protocol)
        training_config["dinov2_img_size"] = args.dinov2_img_size
        main_print(
            f"   Config: LR={training_config['learning_rate']}, "
            f"weight_decay={training_config['weight_decay']}, "
            f"scheduler={training_config['scheduler']}, "
            f"augmentation={training_config['augmentation']}, "
            f"dinov2_img_size={training_config['dinov2_img_size']}, "
            f"amp={not args.no_amp}, grad_accum_steps={args.grad_accum_steps}",
            args=args,
        )

        try:
            # 获取该模型的transform
            train_tf, val_tf = get_model_transform(model_name, protocol, training_config)

            # 存储五折结果
            fold_results = []

            # 五折交叉验证
            for fold_idx, (train_pids, val_pids) in enumerate(patient_stratified_kfold(patient_data, n_splits=5)):
                main_print(f"\n{'─'*50}", args=args)
                main_print(f"📂 Fold {fold_idx + 1}/5", args=args)
                main_print(f"{'─'*50}", args=args)
                main_print(f"   训练集患者数: {len(train_pids)}", args=args)
                main_print(f"   验证集患者数: {len(val_pids)}", args=args)

                # 构建数据集
                train_dataset, val_dataset, val_patient_ids_per_sample = build_fold_datasets(
                    patient_data, train_pids, val_pids, train_tf, val_tf
                )
                if protocol == "few_shot":
                    train_dataset = build_few_shot_train_dataset(
                        train_dataset, shots_per_class=args.shots, seed=42 + fold_idx
                    )

                main_print(f"   训练集图像数: {len(train_dataset)}", args=args)
                main_print(f"   验证集图像数: {len(val_dataset)}", args=args)

                # 验证：检查验证集中没有增强图
                val_has_aug = any('_aug_' in os.path.basename(p) for p in val_dataset.file_paths)
                if val_has_aug:
                    raise ValueError("❌ 验证集中包含增强图，数据划分有误！")
                main_print("   ✅ 验证集确认无增强图", args=args)

                # 验证：检查训练集和验证集患者无交叉
                overlap = set(train_pids) & set(val_pids)
                if overlap:
                    raise ValueError(f"❌ 训练集和验证集患者有交叉: {overlap}")
                main_print("   ✅ 患者隔离确认无交叉", args=args)

                # DataLoader
                # 注意：num_workers 默认用 8，但可通过环境变量 NUM_WORKERS 覆盖
                # 多卡+spawn 模式下若崩溃，建议设置 NUM_WORKERS=0
                num_workers = int(os.environ.get('NUM_WORKERS', 16))
                if getattr(args, "distributed", False):
                    batch_size = max(1, int(np.ceil(args.batch_size / args.world_size)))
                    train_sampler = DistributedSampler(
                        train_dataset,
                        num_replicas=args.world_size,
                        rank=args.rank,
                        shuffle=True,
                        drop_last=False,
                    )
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        sampler=train_sampler,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=False,
                    )
                    if args.rank == 0:
                        print(
                            f"   DDP global batch_size={args.batch_size}, "
                            f"per-rank batch_size={batch_size}, world_size={args.world_size}, "
                            f"num_workers={num_workers}"
                        )
                else:
                    batch_size = args.batch_size
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=False)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=False)

                # 初始化模型
                if model_name == "clip_prompt":
                    model = get_model(model_name, NUM_CLASSES, class_names=NEW_CLASS_NAMES,
                                      protocol=protocol,
                                      partial_blocks=args.partial_blocks,
                                      prompt_templates=DEFAULT_CLIP_PROMPT_TEMPLATES).to(device)
                else:
                    model = get_model(model_name, NUM_CLASSES,
                                      protocol=protocol,
                                      partial_blocks=args.partial_blocks).to(device)
                trainable_params, total_params = count_parameters(model)
                main_print(f"   Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params / total_params:.2%})", args=args)
            
                # 多卡包装
                if getattr(args, "distributed", False):
                    model = DDP(
                        model,
                        device_ids=[args.local_rank],
                        output_device=args.local_rank,
                        find_unused_parameters=False,
                    )
                    if args.rank == 0:
                        print(f"   Using DistributedDataParallel on {args.world_size} GPU process(es)")
                elif n_gpus > 1:
                    model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
                    print(f"   Using DataParallel with {n_gpus} GPUs")
            
                if not getattr(args, "distributed", False) and n_gpus > 1:
                    print(f"   DataParallel will split batch_size={batch_size} across {n_gpus} GPUs ({batch_size//n_gpus} per GPU), num_workers={num_workers}")

                lr = training_config["learning_rate"]
                weight_decay = training_config["weight_decay"]
                trainable_parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = None
                scheduler = None
                if trainable_parameters:
                    optimizer = optimizer_cls(trainable_parameters, lr=lr, weight_decay=weight_decay)
                    scheduler = build_scheduler(optimizer, training_config, args.epochs)
                    main_print(
                        f"   LR={lr}, Optimizer=AdamW(weight_decay={weight_decay}), "
                        f"Scheduler={training_config['scheduler']}",
                        args=args,
                    )
                else:
                    main_print("   Zero-shot: all parameters frozen; skip optimizer and training.", args=args)

                # 训练（带早停）
                fold_save_path = os.path.join(model_save_dir, f"fold_{fold_idx + 1}")
                os.makedirs(fold_save_path, exist_ok=True)

                # 检查该 Fold 是否已完成（已有最终模型）
                best_model_path = os.path.join(fold_save_path, f"best_model_{model_name}_fold{fold_idx+1}.pth")
                if os.path.exists(best_model_path):
                    main_print(f"   ✅ Fold {fold_idx+1} 已完成（存在 {best_model_path}），跳过...", args=args)
                    continue

                # 检查是否有 checkpoint 可恢复
                checkpoint_dir = os.path.join(fold_save_path, f"{experiment_name}_checkpoints")
                latest_checkpoint = os.path.join(checkpoint_dir, f"latest_checkpoint_{experiment_name}.pth")
                resume_checkpoint = None
                if os.path.exists(latest_checkpoint):
                    main_print(f"   🔄 发现 checkpoint: {latest_checkpoint}", args=args)
                    main_print(f"   🔄 自动恢复训练...", args=args)
                    resume_checkpoint = latest_checkpoint

                if protocol == "zero_shot":
                    best_auc = float("nan")
                    train_history = {
                        'train_loss': [], 'val_loss': [], 'accuracy': [], 'precision': [],
                        'recall': [], 'f1': [], 'kappa': [], 'auc': [], 'epoch_time': []
                    }
                    stopped_early = False
                else:
                    model, best_auc, train_history, stopped_early = train_and_evaluate(
                        model, train_loader, val_loader, num_epochs=args.epochs, criterion=criterion,
                        optimizer=optimizer, device=device, save_path=fold_save_path,
                        model_name=experiment_name, early_stop_patience=24,
                        resume_from_checkpoint=resume_checkpoint, scheduler=scheduler,
                        scheduler_name=training_config["scheduler"],
                        use_amp=not args.no_amp,
                        grad_accum_steps=args.grad_accum_steps,
                        rank=args.rank,
                        distributed=getattr(args, "distributed", False),
                    )

                if args.rank == 0:
                    print(f"   Fold {fold_idx + 1} 最佳 AUC: {best_auc:.4f}, 早停: {stopped_early}")

                # ==================== 验证集评估 ====================
                if getattr(args, "distributed", False) and args.rank != 0:
                    if dist.is_initialized():
                        dist.barrier()
                    del model, optimizer, train_loader, val_loader
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                model.eval()
                all_preds, all_labels, all_probs, all_logits = [], [], [], []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        with torch.amp.autocast("cuda", enabled=(not args.no_amp and device.type == "cuda")):
                            outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        probs = torch.softmax(outputs.float(), dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                        all_logits.extend(outputs.cpu().numpy())

                all_labels = np.array(all_labels)
                all_preds = np.array(all_preds)
                all_probs = normalize_probabilities(all_probs)
                all_logits = np.array(all_logits)
                calibrated_probs, temperature = fit_temperature_scaling(all_logits, all_labels, device)

                # ---------- 图像级指标 ----------
                image_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
                image_acc = accuracy_score(all_labels, all_preds)
                image_prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
                image_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
                image_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                image_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                image_kappa = cohen_kappa_score(all_labels, all_preds)
            
                # 计算总体敏感性和特异性（宏平均）
                sensitivities, specificities = compute_sensitivity_specificity(all_labels, all_preds, NUM_CLASSES)
                image_sensitivity = np.mean(sensitivities)
                image_specificity = np.mean(specificities)

                # 各类别指标
                per_class = compute_per_class_metrics(all_labels, all_preds, NEW_CLASS_NAMES)

                # 校准指标
                raw_ece = compute_ece(all_labels, all_probs, n_bins=15)
                raw_brier = compute_brier_score(all_labels, all_probs, NUM_CLASSES)
                ece = compute_ece(all_labels, calibrated_probs, n_bins=15)
                brier = compute_brier_score(all_labels, calibrated_probs, NUM_CLASSES)

                print(f"\n   📊 图像级指标:")
                print(f"      AUC: {image_auc:.4f} | Acc: {image_acc:.4f} | Prec: {image_prec:.4f} | "
                      f"Recall: {image_recall:.4f} |F1: {image_f1:.4f} |macro_F1 :{image_f1_macro:.4f}|Kappa: {image_kappa:.4f}")
                print(f"      Sensitivity: {image_sensitivity:.4f} | Specificity: {image_specificity:.4f}")
                print(f"      Raw ECE: {raw_ece:.4f} | Raw Brier: {raw_brier:.4f}")
                print(f"      Calibrated ECE: {ece:.4f} | Calibrated Brier: {brier:.4f} | Temperature: {temperature:.4f}")
                print(f"   📊 各类别指标:")
                for cls_name, metrics in per_class.items():
                    print(f"      {cls_name}: Prec={metrics['precision']:.4f}, Sens={metrics['sensitivity']:.4f}, "
                          f"Spec={metrics['specificity']:.4f}, F1={metrics['f1-score']:.4f}, Support={metrics['support']}")

                # ---------- 患者级指标 ----------
                patient_acc, patient_cm, patient_true, patient_pred, patient_probs = patient_level_evaluation(
                    val_patient_ids_per_sample, all_labels, calibrated_probs
                )
                print(f"\n   👤 患者级指标:")
                print(f"      Accuracy: {patient_acc:.4f}")
                print(f"      患者数: {len(patient_true)}")

                # ---------- 保存模型 ----------
                torch.save(unwrap_model(model).state_dict(), os.path.join(fold_save_path, f"best_model_{model_name}_fold{fold_idx+1}.pth"))

                # ---------- 保存指标 ----------
                fold_metrics = {
                    'fold': fold_idx + 1,
                    'experiment': {
                        'model_name': model_name,
                        'experiment_name': experiment_name,
                        'protocol': protocol,
                        'training_config': training_config,
                        'shots_per_class': args.shots if protocol == "few_shot" else None,
                        'partial_blocks': args.partial_blocks if protocol == "partial_finetune" else None,
                        'adaptation': {
                            'zero_shot': protocol == "zero_shot",
                            'linear_probe': protocol == "linear_probe",
                            'partial_finetune': protocol == "partial_finetune",
                            'full_finetune': protocol in ["full_finetune", "few_shot"],
                            'trainable_parameters': int(trainable_params),
                            'total_parameters': int(total_params),
                            'trainable_ratio': float(trainable_params / total_params),
                        },
                        'clip_prompt_templates': DEFAULT_CLIP_PROMPT_TEMPLATES if model_name == "clip_prompt" else None,
                        'clip_class_text': CLIP_CLASS_TEXT if model_name == "clip_prompt" else None,
                    },
                    'image_level': {
                        'auc': float(image_auc),
                        'accuracy': float(image_acc),
                        'precision': float(image_prec),
                        'recall': float(image_recall),
                        'sensitivity': float(image_sensitivity),
                        'specificity': float(image_specificity),
                        'f1': float(image_f1),
                        'f1_macro': float(image_f1_macro),
                        'kappa': float(image_kappa),
                        'raw_ece': float(raw_ece),
                        'raw_brier_score': float(raw_brier),
                        'ece': float(ece),
                        'brier_score': float(brier),
                        'temperature': float(temperature),
                        'per_class': per_class
                    },
                    'patient_level': {
                        'accuracy': float(patient_acc),
                        'num_patients': len(patient_true)
                    },
                    'train_history': train_history,
                    'stopped_early': stopped_early,
                    'best_auc': float(best_auc)
                }
                with open(os.path.join(fold_save_path, f"metrics_fold{fold_idx+1}.json"), 'w', encoding='utf-8') as f:
                    json.dump(fold_metrics, f, ensure_ascii=False, indent=2)

                fold_results.append(fold_metrics)

                # 清理显存
                del model, optimizer, train_loader, val_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                if getattr(args, "distributed", False) and dist.is_initialized():
                    dist.barrier()

            if getattr(args, "distributed", False) and args.rank != 0:
                continue

            # ==================== 五折汇总 ====================
            print(f"\n{'='*60}")
            print(f"📊 {model_name} 五折汇总")
            print(f"{'='*60}")

            # 计算平均值
            image_aucs = [r['image_level']['auc'] for r in fold_results]
            image_accs = [r['image_level']['accuracy'] for r in fold_results]
            image_f1s = [r['image_level']['f1'] for r in fold_results]
            image_f1_macros = [r['image_level']['f1_macro'] for r in fold_results]
            image_kappas = [r['image_level']['kappa'] for r in fold_results]
            image_sens = [r['image_level']['sensitivity'] for r in fold_results]
            image_specs = [r['image_level']['specificity'] for r in fold_results]
            image_raw_eces = [r['image_level']['raw_ece'] for r in fold_results]
            image_raw_briers = [r['image_level']['raw_brier_score'] for r in fold_results]
            image_eces = [r['image_level']['ece'] for r in fold_results]
            image_briers = [r['image_level']['brier_score'] for r in fold_results]
            temperatures = [r['image_level']['temperature'] for r in fold_results]
            patient_accs = [r['patient_level']['accuracy'] for r in fold_results]

            # 计算各类别平均指标
            avg_per_class = {}
            for cls_name in NEW_CLASS_NAMES:
                cls_precisions = [r['image_level']['per_class'][cls_name]['precision'] for r in fold_results]
                cls_recalls = [r['image_level']['per_class'][cls_name]['recall'] for r in fold_results]
                cls_sensitivities = [r['image_level']['per_class'][cls_name]['sensitivity'] for r in fold_results]
                cls_specificities = [r['image_level']['per_class'][cls_name]['specificity'] for r in fold_results]
                cls_f1s = [r['image_level']['per_class'][cls_name]['f1-score'] for r in fold_results]
                cls_supports = [r['image_level']['per_class'][cls_name]['support'] for r in fold_results]
                avg_per_class[cls_name] = {
                    'precision_mean': float(np.mean(cls_precisions)),
                    'precision_std': float(np.std(cls_precisions)),
                    'recall_mean': float(np.mean(cls_recalls)),
                    'recall_std': float(np.std(cls_recalls)),
                    'sensitivity_mean': float(np.mean(cls_sensitivities)),
                    'sensitivity_std': float(np.std(cls_sensitivities)),
                    'specificity_mean': float(np.mean(cls_specificities)),
                    'specificity_std': float(np.std(cls_specificities)),
                    'f1_mean': float(np.mean(cls_f1s)),
                    'f1_std': float(np.std(cls_f1s)),
                    'support_mean': float(np.mean(cls_supports)),
                    'support_std': float(np.std(cls_supports)),
                }

            summary = {
                'model_name': model_name,
                'experiment_name': experiment_name,
                'protocol': protocol,
                'training_config': training_config,
                'shots_per_class': args.shots if protocol == "few_shot" else None,
                'partial_blocks': args.partial_blocks if protocol == "partial_finetune" else None,
                'clip_prompt_templates': DEFAULT_CLIP_PROMPT_TEMPLATES if model_name == "clip_prompt" else None,
                'clip_class_text': CLIP_CLASS_TEXT if model_name == "clip_prompt" else None,
                'num_folds': 5,
                'image_level': {
                    'auc_mean': float(np.mean(image_aucs)),
                    'auc_std': float(np.std(image_aucs)),
                    'accuracy_mean': float(np.mean(image_accs)),
                    'accuracy_std': float(np.std(image_accs)),
                    'f1_mean': float(np.mean(image_f1s)),
                    'f1_std': float(np.std(image_f1s)),
                    'f1_macro_mean': float(np.mean(image_f1_macros)),
                    'f1_macro_std': float(np.std(image_f1_macros)),
                    'kappa_mean': float(np.mean(image_kappas)),
                    'kappa_std': float(np.std(image_kappas)),
                    'sensitivity_mean': float(np.mean(image_sens)),
                    'sensitivity_std': float(np.std(image_sens)),
                    'specificity_mean': float(np.mean(image_specs)),
                    'specificity_std': float(np.std(image_specs)),
                    'raw_ece_mean': float(np.mean(image_raw_eces)),
                    'raw_ece_std': float(np.std(image_raw_eces)),
                    'raw_brier_mean': float(np.mean(image_raw_briers)),
                    'raw_brier_std': float(np.std(image_raw_briers)),
                    'ece_mean': float(np.mean(image_eces)),
                    'ece_std': float(np.std(image_eces)),
                    'brier_mean': float(np.mean(image_briers)),
                    'brier_std': float(np.std(image_briers)),
                    'temperature_mean': float(np.mean(temperatures)),
                    'temperature_std': float(np.std(temperatures)),
                    'per_class_avg': avg_per_class,
                },
                'patient_level': {
                    'accuracy_mean': float(np.mean(patient_accs)),
                    'accuracy_std': float(np.std(patient_accs)),
                },
                'fold_results': fold_results
            }

            print(f"   图像级 AUC: {summary['image_level']['auc_mean']:.4f} ± {summary['image_level']['auc_std']:.4f}")
            print(f"   图像级 Acc: {summary['image_level']['accuracy_mean']:.4f} ± {summary['image_level']['accuracy_std']:.4f}")
            print(f"   图像级 F1 (weighted):  {summary['image_level']['f1_mean']:.4f} ± {summary['image_level']['f1_std']:.4f}")
            print(f"   图像级 F1 (macro):     {summary['image_level']['f1_macro_mean']:.4f} ± {summary['image_level']['f1_macro_std']:.4f}")
            print(f"   图像级 Sensitivity: {summary['image_level']['sensitivity_mean']:.4f} ± {summary['image_level']['sensitivity_std']:.4f}")
            print(f"   图像级 Specificity: {summary['image_level']['specificity_mean']:.4f} ± {summary['image_level']['specificity_std']:.4f}")
            print(f"   图像级 Raw ECE: {summary['image_level']['raw_ece_mean']:.4f} ± {summary['image_level']['raw_ece_std']:.4f}")
            print(f"   图像级 Raw Brier: {summary['image_level']['raw_brier_mean']:.4f} ± {summary['image_level']['raw_brier_std']:.4f}")
            print(f"   图像级 Calibrated ECE: {summary['image_level']['ece_mean']:.4f} ± {summary['image_level']['ece_std']:.4f}")
            print(f"   图像级 Calibrated Brier: {summary['image_level']['brier_mean']:.4f} ± {summary['image_level']['brier_std']:.4f}")
            print(f"   Temperature: {summary['image_level']['temperature_mean']:.4f} ± {summary['image_level']['temperature_std']:.4f}")
            print(f"   患者级 Acc: {summary['patient_level']['accuracy_mean']:.4f} ± {summary['patient_level']['accuracy_std']:.4f}")
            print(f"\n   📊 各类别平均指标:")
            for cls_name, metrics in avg_per_class.items():
                print(f"      {cls_name}: Prec={metrics['precision_mean']:.4f}±{metrics['precision_std']:.4f}, "
                      f"Rec={metrics['recall_mean']:.4f}±{metrics['recall_std']:.4f}, "
                      f"Sens={metrics['sensitivity_mean']:.4f}±{metrics['sensitivity_std']:.4f}, "
                      f"Spec={metrics['specificity_mean']:.4f}±{metrics['specificity_std']:.4f}, "
                      f"F1={metrics['f1_mean']:.4f}±{metrics['f1_std']:.4f}, "
                      f"Support={metrics['support_mean']:.1f}")

            # 保存汇总
            with open(os.path.join(model_save_dir, "summary_metrics.json"), 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        except Exception as e:
            error_text = traceback.format_exc()
            print(f"❌ 实验失败，跳过后续 fold 并继续下一个模型: {experiment_name}")
            print(error_text)
            os.makedirs(model_save_dir, exist_ok=True)
            error_record = {
                'model_name': model_name,
                'experiment_name': experiment_name,
                'protocol': protocol,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': error_text,
            }
            with open(os.path.join(model_save_dir, f"error_{experiment_name}.json"), 'w', encoding='utf-8') as f:
                json.dump(error_record, f, ensure_ascii=False, indent=2)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            continue

    main_print(f"\n{'='*60}", args=args)
    main_print("🎉 所有模型训练完成！", args=args)
    main_print(f"{'='*60}", args=args)
    cleanup_distributed_mode(args)


if __name__ == '__main__':
    main()
