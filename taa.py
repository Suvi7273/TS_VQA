# FIXED Module 5: Task-Aware Adapter - Tensor Shape Issue Resolved

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for parameter-efficient fine-tuning
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super(LoRALayer, self).__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x, base_output):
        """
        Args:
            x: input tensor [B, ..., in_features]
            base_output: output from frozen base layer [B, ..., out_features]
            
        Returns:
            adapted_output: base_output + LoRA adaptation
        """
        # LoRA forward pass: x @ A @ B
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return base_output + lora_output

class AdapterLayer(nn.Module):
    """
    Adapter layer for task-specific fine-tuning
    """
    def __init__(self, d_model=256, bottleneck_dim=64, dropout=0.1):
        super(AdapterLayer, self).__init__()
        
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        
        # Down-projection
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        
        # Activation
        self.activation = nn.ReLU()
        
        # Up-projection  
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights to near-zero for stable training
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x):
        """
        Args:
            x: input tensor [B, ..., d_model]
            
        Returns:
            adapted_output: x + adapter(x)
        """
        residual = x
        
        # Adapter forward pass
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        
        # Residual connection
        x = residual + x
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x

class TaskSpecificHead(nn.Module):
    """
    Task-specific prediction heads with adapters - FIXED DIMENSIONS
    """
    def __init__(self, d_model=256, task_dim=None, adapter_dim=64, use_lora=True):
        super(TaskSpecificHead, self).__init__()
        
        self.d_model = d_model
        self.task_dim = task_dim if task_dim is not None else d_model
        self.use_lora = use_lora
        
        # Base layers (frozen during adapter training)
        self.base_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, self.task_dim)
        )
        
        # Adapter layers
        self.adapter = AdapterLayer(d_model, adapter_dim)
        
        # LoRA layers (if enabled)
        if use_lora:
            self.lora1 = LoRALayer(d_model, d_model, rank=8)
            self.lora2 = LoRALayer(d_model, self.task_dim, rank=8)
        
    def forward(self, x, use_adapter=True):
        """
        Args:
            x: input tensor [B, N, d_model]
            use_adapter: whether to use adapter layers
            
        Returns:
            task_output: task-specific predictions
        """
        if use_adapter:
            # Apply adapter first
            x = self.adapter(x)
        
        if self.use_lora and use_adapter:
            # Apply LoRA-enhanced base layers
            base_x = self.base_layers[0](x)  # First linear layer
            x = self.lora1(x, base_x)  # LoRA adaptation
            
            x = self.base_layers[1](x)  # ReLU
            x = self.base_layers[2](x)  # Dropout
            
            base_output = self.base_layers[3](x)  # Second linear layer
            output = self.lora2(x, base_output)  # LoRA adaptation
        else:
            # Standard forward pass
            output = self.base_layers(x)
        
        return output
    
    def freeze_base_layers(self):
        """Freeze base layers for adapter-only training"""
        for param in self.base_layers.parameters():
            param.requires_grad = False
    
    def unfreeze_base_layers(self):
        """Unfreeze base layers for full fine-tuning"""
        for param in self.base_layers.parameters():
            param.requires_grad = True

class MultiTaskAdapter(nn.Module):
    """
    Multi-task adapter that can switch between different tasks
    """
    def __init__(self, d_model=256, num_tasks=5, adapter_dim=64):
        super(MultiTaskAdapter, self).__init__()
        
        self.d_model = d_model
        self.num_tasks = num_tasks
        self.adapter_dim = adapter_dim
        
        # Task-specific adapters
        self.task_adapters = nn.ModuleList([
            AdapterLayer(d_model, adapter_dim)
            for _ in range(num_tasks)
        ])
        
        # Task selection network
        self.task_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_tasks),
            nn.Softmax(dim=-1)
        )
        
        # Shared adapter (always active)
        self.shared_adapter = AdapterLayer(d_model, adapter_dim)
        
    def forward(self, x, task_id=None, use_task_selection=True):
        """
        Args:
            x: input tensor [B, N, d_model]
            task_id: specific task ID (0 to num_tasks-1)
            use_task_selection: whether to use automatic task selection
            
        Returns:
            adapted_x: task-adapted features
            task_weights: task selection weights (if using automatic selection)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Shared adapter (always applied)
        x = self.shared_adapter(x)
        
        if task_id is not None:
            # Use specific task adapter
            x = self.task_adapters[task_id](x)
            task_weights = None
        elif use_task_selection:
            # Automatic task selection
            task_weights = self.task_selector(x.mean(dim=1))  # [B, num_tasks]
            
            # Apply weighted combination of task adapters
            adapted_outputs = []
            for task_idx in range(self.num_tasks):
                task_adapted = self.task_adapters[task_idx](x)
                adapted_outputs.append(task_adapted)
            
            # Weighted combination
            adapted_stack = torch.stack(adapted_outputs, dim=-1)  # [B, N, d_model, num_tasks]
            task_weights_expanded = task_weights.unsqueeze(1).unsqueeze(-2)  # [B, 1, 1, num_tasks]
            
            x = torch.sum(adapted_stack * task_weights_expanded, dim=-1)  # [B, N, d_model]
        else:
            # No task-specific adaptation
            task_weights = None
        
        return x, task_weights

class TaskAwareAdapter(nn.Module):
    """
    Complete Module 5: Task-Aware Adapter - FIXED TENSOR DIMENSIONS
    Provides parameter-efficient fine-tuning capabilities
    """
    def __init__(self, 
                 d_model=256,
                 num_tasks=5,
                 adapter_dim=64,
                 use_lora=True,
                 use_multi_task=True):
        super(TaskAwareAdapter, self).__init__()
        
        self.d_model = d_model
        self.num_tasks = num_tasks
        self.use_lora = use_lora
        self.use_multi_task = use_multi_task
        
        # Core adapter components
        if use_multi_task:
            self.multi_task_adapter = MultiTaskAdapter(d_model, num_tasks, adapter_dim)
        else:
            self.single_adapter = AdapterLayer(d_model, adapter_dim)
        
        # Task-specific prediction heads with adapters
        self.classification_head = TaskSpecificHead(d_model, 3, adapter_dim, use_lora)  # 3 classes
        self.bbox_head = TaskSpecificHead(d_model, 4, adapter_dim, use_lora)  # 4 bbox coords
        self.polygon_head = TaskSpecificHead(d_model, 16, adapter_dim, use_lora)  # 16 polygon points
        
        # 🔥 FIXED: Text head with correct dimensions
        # Calculate correct text output dimension
        self.text_head = TaskSpecificHead(d_model, 2500, adapter_dim, use_lora)  # 25 * 100 = 2500
        
        # Adapter control
        self.adapter_enabled = True
        
    def forward(self, queries, task_id=None):
        """
        Args:
            queries: [B, N, d_model] input queries from previous modules
            task_id: specific task ID for task-specific adaptation
            
        Returns:
            adapted_queries: [B, N, d_model] task-adapted queries
            adapter_outputs: dict with adapter-specific information
        """
        adapted_queries = queries
        task_weights = None
        
        if self.adapter_enabled:
            if self.use_multi_task:
                adapted_queries, task_weights = self.multi_task_adapter(
                    adapted_queries, task_id=task_id
                )
            else:
                adapted_queries = self.single_adapter(adapted_queries)
        
        # Prepare adapter outputs
        adapter_outputs = {
            'task_weights': task_weights,
            'adapter_enabled': self.adapter_enabled,
            'current_task_id': task_id
        }
        
        return adapted_queries, adapter_outputs
    
    def get_task_specific_predictions(self, adapted_queries, max_text_len=25, vocab_size=100):
        """
        Generate task-specific predictions using adapter-enhanced heads - FIXED DIMENSIONS
        
        Args:
            adapted_queries: [B, N, d_model] adapter-enhanced queries
            max_text_len: maximum text sequence length
            vocab_size: vocabulary size for text predictions
            
        Returns:
            predictions: dict with task-specific predictions
        """
        batch_size, num_queries, _ = adapted_queries.shape
        
        # Task-specific predictions
        pred_logits = self.classification_head(adapted_queries, use_adapter=self.adapter_enabled)
        pred_boxes = self.bbox_head(adapted_queries, use_adapter=self.adapter_enabled).sigmoid()
        pred_polygons = self.polygon_head(adapted_queries, use_adapter=self.adapter_enabled).sigmoid()
        
        # 🔥 FIXED: Text predictions with correct reshaping
        text_features = self.text_head(adapted_queries, use_adapter=self.adapter_enabled)  # [B, N, 2500]
        
        # Reshape correctly: [B, N, 2500] -> [B, N, 25, 100]
        text_logits = text_features.view(batch_size, num_queries, max_text_len, vocab_size)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'pred_polygons': pred_polygons,
            'pred_texts': text_logits
        }
    
    def enable_adapter_mode(self):
        """Enable adapter-only training (freeze base layers)"""
        self.adapter_enabled = True
        
        # Freeze base layers in task-specific heads
        self.classification_head.freeze_base_layers()
        self.bbox_head.freeze_base_layers()
        self.polygon_head.freeze_base_layers()
        self.text_head.freeze_base_layers()
        
    def enable_full_training(self):
        """Enable full model training (unfreeze all layers)"""
        self.adapter_enabled = True
        
        # Unfreeze base layers in task-specific heads
        self.classification_head.unfreeze_base_layers()
        self.bbox_head.unfreeze_base_layers()
        self.polygon_head.unfreeze_base_layers()
        self.text_head.unfreeze_base_layers()
    
    def disable_adapters(self):
        """Disable adapters (use base model only)"""
        self.adapter_enabled = False
    
    def get_adapter_parameters(self):
        """Get only adapter parameters for efficient training"""
        adapter_params = []
        
        # Multi-task adapter parameters
        if self.use_multi_task:
            adapter_params.extend(list(self.multi_task_adapter.parameters()))
        else:
            adapter_params.extend(list(self.single_adapter.parameters()))
        
        # LoRA parameters from task-specific heads
        if self.use_lora:
            for head in [self.classification_head, self.bbox_head, self.polygon_head, self.text_head]:
                if hasattr(head, 'lora1'):
                    adapter_params.extend([head.lora1.lora_A, head.lora1.lora_B])
                if hasattr(head, 'lora2'):
                    adapter_params.extend([head.lora2.lora_A, head.lora2.lora_B])
        
        return adapter_params
    
    def get_parameter_count(self):
        """Get parameter counts for analysis"""
        total_params = sum(p.numel() for p in self.parameters())
        adapter_params = sum(p.numel() for p in self.get_adapter_parameters())
        
        return {
            'total_parameters': total_params,
            'adapter_parameters': adapter_params,
            'reduction_ratio': adapter_params / total_params if total_params > 0 else 0.0
        }
