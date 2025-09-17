# COMPLETE test.py for VimTS with Module 4 (PQGM)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import json
from PIL import Image

# Import loss function
from loss import VimTSLoss

# VimTS Complete Model with Module 5 (Task-Aware Adapter) Integration

import torch
import torch.nn as nn
import torch.nn.functional as F

class VimTSWithTaskAdapter(nn.Module):
    """
    Complete VimTS Model: Modules 1 + 2 + 3 + 4 + 5 + 7
    Now includes Task-Aware Adapter for parameter-efficient fine-tuning
    """
    def __init__(self, 
                 num_classes=2, 
                 vocab_size=100, 
                 max_text_len=25,
                 num_detection_queries=100,
                 num_recognition_queries=25,
                 num_domains=5,
                 num_tasks=5,
                 use_adapter=True):
        super().__init__()
        
        # Module 1: Feature Extraction
        from backbone import VimTSFeatureExtraction
        self.feature_extractor = VimTSFeatureExtraction(pretrained=True)
        
        # Module 2: Query Initialization
        from queryInitialization_CORRECTED import QueryInitialization
        self.query_initializer = QueryInitialization(
            feature_dim=256,
            num_detection_queries=num_detection_queries,
            num_recognition_queries=num_recognition_queries
        )
        
        # Module 3: Decoder
        from module3_decoder import CompleteVimTSDecoder
        self.decoder = CompleteVimTSDecoder(
            d_model=256,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # Module 4: PQGM
        from module4_pqgm import PQGM
        self.pqgm = PQGM(
            d_model=256,
            num_heads=8,
            num_domains=num_domains,
            max_prompt_len=50,
            dropout=0.1
        )
        
        # Module 5: Task-Aware Adapter (NEW!)
        from module5_task_aware_adapter import TaskAwareAdapter
        self.task_adapter = TaskAwareAdapter(
            d_model=256,
            num_tasks=num_tasks,
            adapter_dim=64,
            use_lora=True,
            use_multi_task=True
        )
        
        # Legacy prediction heads (for compatibility when adapters disabled)
        self.legacy_class_head = nn.Linear(256, num_classes + 1)
        self.legacy_bbox_head = nn.Linear(256, 4)
        self.legacy_polygon_head = nn.Linear(256, 16)
        self.legacy_text_head = nn.Linear(256, max_text_len * vocab_size)
        
        # Store parameters
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        self.num_detection_queries = num_detection_queries
        self.num_recognition_queries = num_recognition_queries
        self.num_domains = num_domains
        self.num_tasks = num_tasks
        self.use_adapter = use_adapter
        
    def forward(self, images, domain_id=None, task_id=None, granularity_hints=None):
        """
        Complete forward pass through all modules including Task-Aware Adapter
        
        Args:
            images: [B, 3, H, W] input images
            domain_id: int, domain identifier for PQGM
            task_id: int, task identifier for Task-Aware Adapter
            granularity_hints: [B, N, 3] granularity preferences for PQGM
        """
        batch_size = images.shape[0]
        
        # Module 1: Feature Extraction
        enhanced_features = self.feature_extractor(images)  # [B, 256, H', W']
        
        # Prepare visual features for decoder and PQGM
        B, C, H, W = enhanced_features.shape
        visual_features = enhanced_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Module 2: Query Initialization
        detection_queries, recognition_queries, coarse_preds = self.query_initializer(enhanced_features)
        
        # Combine queries
        all_queries = torch.cat([detection_queries, recognition_queries], dim=1)  # [B, 125, C]
        
        # Module 3: Decoder - Vision-Language Communication
        decoder_queries, attention_weights = self.decoder(all_queries, visual_features)
        
        # Module 4: PQGM - Prompt Query Generation
        pqgm_queries, pqgm_outputs = self.pqgm(
            decoder_queries, 
            visual_features, 
            domain_id=domain_id,
            granularity_hints=granularity_hints,
            training=self.training
        )
        
        # Module 5: Task-Aware Adapter (NEW!)
        if self.use_adapter:
            adapted_queries, adapter_outputs = self.task_adapter(pqgm_queries, task_id=task_id)
            
            # Use adapter-enhanced prediction heads
            predictions = self.task_adapter.get_task_specific_predictions(
                adapted_queries, self.max_text_len, self.vocab_size
            )
        else:
            # Use legacy prediction heads
            adapted_queries = pqgm_queries
            adapter_outputs = {'adapter_enabled': False}
            
            # Get image dimensions for proper scaling  
            _, _, img_h, img_w = images.shape
            max_size = max(img_h, img_w)
            
            predictions = {
                'pred_logits': self.legacy_class_head(adapted_queries),
                'pred_boxes': self.legacy_bbox_head(adapted_queries).sigmoid() * max_size,
                'pred_polygons': self.legacy_polygon_head(adapted_queries).sigmoid() * max_size,
                'pred_texts': self.legacy_text_head(adapted_queries).view(batch_size, -1, self.max_text_len, self.vocab_size)
            }
        
        # Scale predictions if using adapters
        if self.use_adapter:
            _, _, img_h, img_w = images.shape
            max_size = max(img_h, img_w)
            predictions['pred_boxes'] = predictions['pred_boxes'] * max_size
            predictions['pred_polygons'] = predictions['pred_polygons'] * max_size
        
        # Combine all outputs
        return {
            **predictions,
            'coarse_predictions': coarse_preds,
            'attention_weights': attention_weights,
            'pqgm_outputs': pqgm_outputs,
            'adapter_outputs': adapter_outputs  # NEW: Task-Aware Adapter outputs
        }
    
    def enable_adapter_training(self):
        """Enable parameter-efficient adapter training"""
        self.use_adapter = True
        self.task_adapter.enable_adapter_mode()
        
        # Freeze backbone modules for efficient training
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.query_initializer.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        # Keep PQGM trainable as it's also part of adaptation
        
    def enable_full_training(self):
        """Enable full model training"""
        self.use_adapter = True
        self.task_adapter.enable_full_training()
        
        # Unfreeze all parameters
        for param in self.parameters():
            param.requires_grad = True
    
    def disable_adapters(self):
        """Disable adapters and use base model"""
        self.use_adapter = False
        self.task_adapter.disable_adapters()
    
    def get_adapter_parameters(self):
        """Get only adapter parameters for efficient training"""
        if self.use_adapter:
            adapter_params = self.task_adapter.get_adapter_parameters()
            # Also include PQGM parameters for adaptation
            adapter_params.extend(list(self.pqgm.parameters()))
            return adapter_params
        return []
    
    def get_parameter_statistics(self):
        """Get parameter count statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        adapter_params = sum(p.numel() for p in self.get_adapter_parameters())
        
        stats = {
            'total_parameters': total_params,
            'adapter_parameters': adapter_params,
            'backbone_parameters': total_params - adapter_params,
            'adapter_percentage': (adapter_params / total_params) * 100 if total_params > 0 else 0.0,
            'parameter_reduction': ((total_params - adapter_params) / total_params) * 100 if total_params > 0 else 0.0
        }
        
        return stats

# Updated test model with Module 5
class MinimalVimTSModelWithTaskAdapter(nn.Module):
    """Test model with Module 5 (Task-Aware Adapter)"""
    def __init__(self, num_classes=2, vocab_size=100, max_text_len=25, num_domains=5, num_tasks=5):
        super().__init__()
        
        self.vimts_model = VimTSWithTaskAdapter(
            num_classes=num_classes,
            vocab_size=vocab_size,
            max_text_len=max_text_len,
            num_detection_queries=100,
            num_recognition_queries=25,
            num_domains=num_domains,
            num_tasks=num_tasks,
            use_adapter=True
        )
        
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        
    def forward(self, images, domain_id=None, task_id=None, granularity_hints=None):
        return self.vimts_model(images, domain_id, task_id, granularity_hints)
    
    def enable_adapter_training(self):
        self.vimts_model.enable_adapter_training()
    
    def enable_full_training(self):
        self.vimts_model.enable_full_training()
    
    def get_parameter_statistics(self):
        return self.vimts_model.get_parameter_statistics()

# Test function for Module 5
def test_module5_integration():
    """Test Module 5 (Task-Aware Adapter) integration"""
    print("🔍 Testing Module 5: Task-Aware Adapter Integration...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Create test model with Task-Aware Adapter
    model = MinimalVimTSModelWithTaskAdapter(num_domains=5, num_tasks=5).to(device)
    
    # Test with dummy images
    batch_size, channels, height, width = 2, 3, 640, 480
    test_images = torch.randn(batch_size, channels, height, width).to(device)
    
    # Test with task and domain IDs
    domain_id = 2
    task_id = 1
    
    try:
        # Test 1: Full model forward pass
        print("🔧 Testing full model...")
        with torch.no_grad():
            predictions = model(test_images, domain_id=domain_id, task_id=task_id)
        
        print(f"✅ Forward pass successful!")
        print(f"   Images shape: {test_images.shape}")
        print(f"   Domain ID: {domain_id}, Task ID: {task_id}")
        print(f"   Pred logits shape: {predictions['pred_logits'].shape}")
        print(f"   Pred boxes shape: {predictions['pred_boxes'].shape}")
        print(f"   Pred polygons shape: {predictions['pred_polygons'].shape}")
        print(f"   Pred texts shape: {predictions['pred_texts'].shape}")
        
        # Check Module 5 outputs (Task-Aware Adapter)
        if 'adapter_outputs' in predictions:
            adapter = predictions['adapter_outputs']
            print(f"   ✅ Module 5 - Adapter enabled: {adapter['adapter_enabled']}")
            print(f"   ✅ Module 5 - Current task: {adapter['current_task_id']}")
            if adapter['task_weights'] is not None:
                print(f"   ✅ Module 5 - Task weights shape: {adapter['task_weights'].shape}")
            print("   ✅ Module 5 (Task-Aware Adapter) working!")
        
        # Test 2: Parameter efficiency analysis
        print("\n📊 Testing parameter efficiency...")
        stats = model.get_parameter_statistics()
        print(f"   📈 Total parameters: {stats['total_parameters']:,}")
        print(f"   📈 Adapter parameters: {stats['adapter_parameters']:,}")
        print(f"   📈 Adapter percentage: {stats['adapter_percentage']:.2f}%")
        print(f"   📈 Parameter reduction: {stats['parameter_reduction']:.2f}%")
        
        # Test 3: Adapter training mode
        print("\n🎯 Testing adapter training mode...")
        model.enable_adapter_training()
        
        # Count trainable parameters in adapter mode
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   🔧 Trainable parameters in adapter mode: {trainable_params:,}")
        print(f"   🔧 Training efficiency: {(stats['total_parameters'] - trainable_params) / stats['total_parameters'] * 100:.1f}% reduction")
        
        print("✅ Module 5 integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Module 5 integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Efficient training setup for Module 5
def setup_efficient_training(model, learning_rate=1e-4):
    """
    Setup parameter-efficient training with adapters
    
    Args:
        model: VimTS model with Task-Aware Adapter
        learning_rate: learning rate for adapter parameters
        
    Returns:
        optimizer: optimizer for adapter parameters only
        scheduler: learning rate scheduler
    """
    # Enable adapter training mode
    model.enable_adapter_training()
    
    # Get only adapter parameters
    adapter_params = model.get_adapter_parameters()
    
    print(f"🎯 Efficient Training Setup:")
    print(f"   📊 Adapter parameters: {sum(p.numel() for p in adapter_params):,}")
    print(f"   📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   📊 Training efficiency: {len(adapter_params) / len(list(model.parameters())) * 100:.1f}% of parameters")
    
    # Create optimizer for adapter parameters only
    optimizer = torch.optim.AdamW(
        adapter_params, 
        lr=learning_rate, 
        weight_decay=0.01
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=1000,  # Adjust based on training steps
        eta_min=learning_rate * 0.1
    )
    
    return optimizer, scheduler

# ========================================
# Dataset Loading (Same as before)
# ========================================

class VimTSRealDataset(Dataset):
    """Dataset loader for COCO-style annotation format"""
    def __init__(self, dataset_path, split='train', dataset_name='sample'):
        self.dataset_path = dataset_path
        self.split = split
        self.dataset_name = dataset_name
        
        # Paths
        self.annotation_file = os.path.join(dataset_path, dataset_name, f'{split}.json')
        self.image_dir = os.path.join(dataset_path, dataset_name, 'img')
        
        # Load JSON
        with open(self.annotation_file, 'r') as f:
            coco = json.load(f)
        
        # Map image_id → image info
        self.images = {img['id']: img for img in coco['images']}
        self.annotations = coco['annotations']
        
        # Group annotations by image_id
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann['image_id'], []).append(ann)
        
        self.image_ids = list(self.images.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        ann_list = self.image_to_anns.get(image_id, [])
        
        # Load image
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Parse annotations
        labels, boxes, polygons, texts = [], [], [], []
        
        for ann in ann_list:
            labels.append(ann.get('category_id', 1))
            
            # Bounding box [x, y, w, h] → [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Polygon from segmentation (use first if multiple)
            if 'segmentation' in ann and len(ann['segmentation']) > 0:
                poly = np.array(ann['segmentation'][0]).reshape(-1, 2)
                polygon_flat = poly.flatten()[:16]
                if len(polygon_flat) < 16:
                    polygon_flat = np.pad(polygon_flat, (0, 16 - len(polygon_flat)))
            else:
                polygon_flat = np.zeros(16)
            polygons.append(polygon_flat)
            
            # Text tokens (from `rec` if present, else empty)
            text_tokens = ann.get('rec', [])
            text_tokens = self.text_to_tokens(text_tokens)
            texts.append(text_tokens)
        
        target = {
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float),
            'polygons': torch.tensor(polygons, dtype=torch.float),
            'texts': torch.tensor(texts, dtype=torch.long)
        }
        
        return image, target
    
    def text_to_tokens(self, rec_field, max_len=25, vocab_size=100):
        """Convert 'rec' field or string to tokens"""
        if isinstance(rec_field, str):
            tokens = [min(ord(c), vocab_size - 1) for c in rec_field[:max_len]]
        elif isinstance(rec_field, list):  # already numeric
            tokens = [min(int(v), vocab_size - 1) for v in rec_field[:max_len]]
        else:
            tokens = []
        
        # Replace 96 (unused/pad in your JSON) with 0
        tokens = [0 if t == 96 else t for t in tokens]
        tokens += [0] * (max_len - len(tokens))
        return tokens[:max_len]

def create_real_dataloader(dataset_path):
    """Create DataLoader with real VimTS datasets"""
    
    train_dataset = VimTSRealDataset(
        dataset_path=dataset_path,
        split='train',
        dataset_name='sample'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return train_loader

def collate_fn(batch):
    images, targets = zip(*batch)
    
    # find max height and width
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    padded_images = []
    for img in images:
        c, h, w = img.shape
        padded = torch.zeros((c, max_h, max_w))
        padded[:, :h, :w] = img
        padded_images.append(padded)
    
    images = torch.stack(padded_images, dim=0)
    return images, list(targets)


def test_module4_integration():
    """Test Module 4 (PQGM) integration with synthetic data"""
    print("🔍 Testing Module 4: PQGM Integration...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Create test model with PQGM
    model = MinimalVimTSModelWithPQGM(num_domains=5).to(device)
    
    # Test with dummy images
    batch_size, channels, height, width = 2, 3, 640, 480
    test_images = torch.randn(batch_size, channels, height, width).to(device)
    
    # Optional: test with domain and granularity hints
    domain_id = 1  # Simulate domain 1
    granularity_hints = torch.rand(batch_size, 125, 3).to(device)  # Random granularity preferences
    granularity_hints = F.softmax(granularity_hints, dim=-1)
    
    try:
        with torch.no_grad():
            predictions = model(test_images, domain_id=domain_id, granularity_hints=granularity_hints)
        
        print(f"✅ Forward pass successful!")
        print(f"   Images shape: {test_images.shape}")
        print(f"   Pred logits shape: {predictions['pred_logits'].shape}")
        print(f"   Pred boxes shape: {predictions['pred_boxes'].shape}")
        print(f"   Pred polygons shape: {predictions['pred_polygons'].shape}")
        print(f"   Pred texts shape: {predictions['pred_texts'].shape}")
        
        # Check Module 4 outputs (PQGM) - NEW!
        if 'pqgm_outputs' in predictions:
            pqgm = predictions['pqgm_outputs']
            print(f"   ✅ Module 4 - Prompt weights shape: {pqgm['prompt_weights'].shape}")
            print(f"   ✅ Module 4 - Granularity dist: {pqgm['granularity_distribution'].shape}")
            if pqgm['domain_logits'] is not None:
                print(f"   ✅ Module 4 - Domain logits: {pqgm['domain_logits'].shape}")
            print("   ✅ Module 4 (PQGM) working!")
            
        print("✅ Module 4 integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Module 4 integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test if gradients flow properly through the model"""
    print("🔍 Testing gradient flow...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use PQGM-enhanced model
    model = MinimalVimTSModelWithPQGM(num_domains=5).to(device)
    criterion = VimTSLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy batch
    images = torch.randn(2, 3, 640, 640).to(device)
    targets = [
        {
            'labels': torch.randint(0, 2, (3,)).to(device),
            'boxes': torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150], [200, 200, 250, 250]], dtype=torch.float).to(device),
            'polygons': torch.rand(3, 16).to(device) * 640,
            'texts': torch.randint(0, 100, (3, 25)).to(device)
        },
        {
            'labels': torch.randint(0, 2, (2,)).to(device),
            'boxes': torch.tensor([[20, 20, 80, 80], [300, 300, 400, 400]], dtype=torch.float).to(device),
            'polygons': torch.rand(2, 16).to(device) * 640,
            'texts': torch.randint(0, 100, (2, 25)).to(device)
        }
    ]
    
    try:
        # Forward pass with domain ID
        predictions = model(images, domain_id=2)
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"✅ Gradient norm: {grad_norm:.4f}")
        
        optimizer.step()
        print("✅ Gradient flow test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ========================================
# Main Execution
# ========================================

if __name__ == "__main__":
    dataset_path = r"/content/drive/MyDrive"  # Update this path
    
    print("🎯 Testing VimTS with Module 4 (PQGM)")
    print("=" * 50)
    
    # Test 1: Module 4 integration with synthetic data
    success = test_module4_integration()
    
    if success:
        # Test 2: Real dataset with Module 4
        success = dry_run_test_with_module4(dataset_path)
    
    if success:
        # Test 3: Gradient flow
        success = test_gradient_flow()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! Your VimTS with Module 4 (PQGM) is working correctly!")
        print("🚀 You now have a state-of-the-art text spotter with:")
        print("   ✅ Module 1: Feature Extraction")
        print("   ✅ Module 2: Query Initialization")
        print("   ✅ Module 3: Decoder")
        print("   ✅ Module 4: PQGM (Prompt Query Generation)")
        print("   ✅ Module 7: Loss Function")
        print("\n🎯 Ready for Module 5: Task-Aware Adapter or deployment!")
    else:
        print("\n❌ Tests failed. Please fix the issues before proceeding.")

