#!/usr/bin/env python3
"""
断点续训管理器
支持保存和加载完整的训练状态，包括模型权重、优化器状态、epoch信息等
"""

import os
import torch
import json
import time
from typing import Dict, Any, Optional, Tuple

class CheckpointManager:
    """
    断点续训管理器
    - 只保存 latest 和 best
    - best 由外部传入的 best_metric 决定（如 AUC）
    """

    def __init__(self, save_dir: str, model_name: str):
        self.save_dir = save_dir
        self.model_name = model_name
        self.checkpoint_dir = os.path.join(save_dir, f"{model_name}_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True, mode=0o755)

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        best_metric: float = 0.0,          # ⭐ 改名：与 train_and_evaluate 对齐
        train_history: Optional[Dict] = None,
        config: Optional[Dict] = None,
        is_best: bool = False
    ) -> str:
        """
        保存 checkpoint
        - best_metric: 当前为止的最优指标（例如 Val AUC）
        """

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric,     # ⭐ 统一字段名
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if train_history is not None:
            checkpoint['train_history'] = train_history
        if config is not None:
            checkpoint['config'] = config

        # ---------- latest ----------
        latest_checkpoint = os.path.join(
            self.checkpoint_dir,
            f"latest_checkpoint_{self.model_name}.pth"
        )
        torch.save(checkpoint, latest_checkpoint)

        # ---------- best ----------
        if is_best:
            best_checkpoint = os.path.join(
                self.checkpoint_dir,
                f"best_checkpoint_{self.model_name}.pth"
            )
            torch.save(checkpoint, best_checkpoint)
            print(f"🏆 新的最佳模型已保存 (Best Metric: {best_metric:.4f})")

        return latest_checkpoint

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
        load_best: bool = False
    ) -> Tuple[Dict, int, float]:
        """
        返回:
        - checkpoint
        - epoch
        - best_metric
        """

        if checkpoint_path is None:
            filename = "best_checkpoint" if load_best else "latest_checkpoint"
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{filename}_{self.model_name}.pth"
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint 文件不存在: {checkpoint_path}")

        print(f"🔄 正在加载 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        epoch = checkpoint.get("epoch", 0)
        best_metric = checkpoint.get("best_metric", float('-inf'))

        print("✅ Checkpoint 加载成功:")
        print(f"   - Epoch: {epoch}")
        print(f"   - Best Metric: {best_metric:.4f}")
        print(f"   - 保存时间: {checkpoint.get('timestamp', 'Unknown')}")

        return checkpoint, epoch, best_metric


    
    def load_model_state(self, model: torch.nn.Module, checkpoint_path: Optional[str] = None, device: str = 'cpu') -> torch.nn.Module:
        """只加载模型权重"""
        checkpoint, _, _ = self.load_checkpoint(checkpoint_path, device)
        
        # 处理DDP模型的情况
        model_state = checkpoint['model_state_dict']
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
            
        return model
    
    def load_optimizer_state(self, optimizer: torch.optim.Optimizer, checkpoint_path: Optional[str] = None, device: str = 'cpu') -> torch.optim.Optimizer:
        """只加载优化器状态"""
        checkpoint, _, _ = self.load_checkpoint(checkpoint_path, device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return optimizer
    
    def resume_training(self,
                   model,
                   optimizer,
                   scheduler=None,
                   checkpoint_path=None,
                   device='cpu'):

        checkpoint, epoch, mean_metric = self.load_checkpoint(checkpoint_path, device)

        # load model
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        # load optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # load scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        history = checkpoint.get("train_history", None)

        print(f"🚀 训练已恢复，从第 {epoch + 1} 个 epoch 开始")
        return epoch, mean_metric, history

    
    def list_checkpoints(self) -> Dict[str, str]:
        """列出所有可用的checkpoint"""
        checkpoints = {}
        
        # 查找所有checkpoint文件
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(f"checkpoint_{self.model_name}") and file.endswith(".pth"):
                file_path = os.path.join(self.checkpoint_dir, file)
                checkpoints[file] = file_path
                
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """清理旧的checkpoint，只保留最新的N个"""
        checkpoints = self.list_checkpoints()
        
        # 按文件名排序（包含epoch信息）
        sorted_checkpoints = sorted(checkpoints.items())
        
        # 删除旧的checkpoint
        if len(sorted_checkpoints) > keep_last_n:
            for file, path in sorted_checkpoints[:-keep_last_n]:
                if not file.startswith("best_checkpoint") and not file.startswith("latest_checkpoint"):
                    os.remove(path)
                    print(f"🗑️  已删除旧checkpoint: {file}")

