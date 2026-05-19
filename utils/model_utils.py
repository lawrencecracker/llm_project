"""
Model utility functions for parameter counting and analysis
"""

import torch
import logging

logger = logging.getLogger(__name__)


def count_parameters(model):
    """Count total, trainable, and frozen parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_ratio": (trainable / total) * 100 if total > 0 else 0,
    }


def print_model_params(model):
    """Print model parameter statistics"""
    params = count_parameters(model)
    
    print("\n" + "="*50)
    print("MODEL PARAMETER STATISTICS")
    print("="*50)
    print(f"Total Parameters:     {params['total']:>15,}")
    print(f"Trainable:            {params['trainable']:>15,}")
    print(f"Frozen:               {params['frozen']:>15,}")
    print(f"Trainable Ratio:      {params['trainable_ratio']:>15.2f}%")
    print("="*50 + "\n")


def get_model_size(model):
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def estimate_memory_usage(model, batch_size, seq_length):
    """Estimate GPU memory usage during training"""
    model_size = get_model_size(model)
    estimated_memory = model_size * 2 + (batch_size * seq_length * 4 * 32 / 1024 / 1024)
    return estimated_memory
