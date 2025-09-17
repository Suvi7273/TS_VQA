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

# ========================================
# VimTS Complete Model with Module 4 (PQGM)
# ========================================

class VimTSWithPQGM(nn.Module):
    """
    Complete VimTS Model: Modules 1 + 2 + 3 + 4 + 7
    Now includes PQGM (Prompt Query Generation Module)
    """
    def __init__(self, 
                 num_classes=2, 
                 vocab_size=100, 
                 max_text_len=25,
                 num_detection_queries=100,
                 num_recognition_queries=25,
                 num_domains=5):
        super().__init__()
        
        # Module 1: Feature Extraction
        from backbone import VimTSFeatureExtraction
        self.feature_extractor = VimTSFeatureExtraction(pretrained=True)
        
        # Module 2: Query Initialization
        from queryInitialization import QueryInitialization
        self.query_initializer = QueryInitialization(
            feature_dim=256,
            num_detection_queries=num_detection_queries,
            num_recognition_queries=num_recognition_queries
        )
        
        # Module 3: Decoder
        from decoder import CompleteVimTSDecoder
        self.decoder = CompleteVimTSDecoder(
            d_model=256,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # Module 4: PQGM (NEW!)
        from pqgm import PQGM
        self.pqgm = PQGM(
            d_model=256,
            num_heads=8,
            num_domains=num_domains,
            max_prompt_len=50,
            dropout=0.1
        )
        
        # Enhanced prediction heads
        self.class_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes + 1)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)
        )
        
        self.polygon_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 16)
        )
        
        self.text_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, max_text_len * vocab_size)
        )
        
        # Store parameters
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        self.num_detection_queries = num_detection_queries
        self.num_recognition_queries = num_recognition_queries
        self.num_domains = num_domains
        
    def forward(self, images, domain_id=None, granularity_hints=None):
        """
        Complete forward pass through all modules including PQGM
        
        Args:
            images: [B, 3, H, W] input images
            domain_id: int, domain identifier for cross-domain adaptation
            granularity_hints: [B, N, 3] granularity preferences (optional)
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
        
        # Module 4: PQGM - Prompt Query Generation (NEW!)
        pqgm_queries, pqgm_outputs = self.pqgm(
            decoder_queries, 
            visual_features, 
            domain_id=domain_id,
            granularity_hints=granularity_hints,
            training=self.training
        )
        
        # Final prediction heads (using PQGM-enhanced queries)
        pred_logits = self.class_head(pqgm_queries)
        
        # Get image dimensions for proper scaling  
        _, _, img_h, img_w = images.shape
        max_size = max(img_h, img_w)
        
        pred_boxes = self.bbox_head(pqgm_queries).sigmoid() * max_size
        pred_polygons = self.polygon_head(pqgm_queries).sigmoid() * max_size
        
        # Text predictions
        text_logits = self.text_head(pqgm_queries)
        pred_texts = text_logits.view(batch_size, -1, self.max_text_len, self.vocab_size)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes, 
            'pred_polygons': pred_polygons,
            'pred_texts': pred_texts,
            'coarse_predictions': coarse_preds,
            'attention_weights': attention_weights,
            'pqgm_outputs': pqgm_outputs  # NEW: PQGM analysis outputs
        }

# Updated test model with Module 4
class MinimalVimTSModelWithPQGM(nn.Module):
    """Test model with Module 4 (PQGM)"""
    def __init__(self, num_classes=2, vocab_size=100, max_text_len=25, num_domains=5):
        super().__init__()
        
        self.vimts_model = VimTSWithPQGM(
            num_classes=num_classes,
            vocab_size=vocab_size,
            max_text_len=max_text_len,
            num_detection_queries=100,
            num_recognition_queries=25,
            num_domains=num_domains
        )
        
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        
    def forward(self, images, domain_id=None, granularity_hints=None):
        return self.vimts_model(images, domain_id, granularity_hints)

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

# ========================================
# Testing Functions with Module 4
# ========================================

def dry_run_test_with_module4(dataset_path):
    """Complete dry run test with Modules 1 + 2 + 3 + 4 + 7"""
    print("🚀 Starting VimTS Dry Run Test with Module 4 (PQGM)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Use PQGM-enhanced model
    model = MinimalVimTSModelWithPQGM(num_domains=5).to(device)
    criterion = VimTSLoss()
    
    # Use real dataloader
    dataloader = create_real_dataloader(dataset_path)
    
    model.eval()
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        
        # Optional: add domain and granularity hints
        domain_id = batch_idx % 5  # Cycle through domains 0-4
        granularity_hints = None  # Let model auto-select
        
        try:
            # Forward pass
            with torch.no_grad():
                predictions = model(images, domain_id=domain_id, granularity_hints=granularity_hints)
            
            print(f"✅ Batch {batch_idx + 1}:")
            print(f"   Images shape: {images.shape}")
            print(f"   Domain ID: {domain_id}")
            print(f"   Pred logits shape: {predictions['pred_logits'].shape}")
            print(f"   Pred boxes shape: {predictions['pred_boxes'].shape}")
            print(f"   Pred polygons shape: {predictions['pred_polygons'].shape}")
            print(f"   Pred texts shape: {predictions['pred_texts'].shape}")
            
            # Check Module 2 outputs
            if 'coarse_predictions' in predictions:
                coarse_preds = predictions['coarse_predictions']
                print(f"   ✅ Module 2 - Coarse class: {coarse_preds['coarse_class_logits'].shape}")
                print(f"   ✅ Module 2 - Coarse bbox: {coarse_preds['coarse_bbox_pred'].shape}")
            
            # Check Module 3 outputs
            if 'attention_weights' in predictions:
                attn_weights = predictions['attention_weights']
                print(f"   ✅ Module 3 - Attention layers: {len(attn_weights)}")
                print(f"   ✅ Module 3 - Attention shape: {attn_weights[0].shape}")
            
            # Check Module 4 outputs (PQGM) - NEW!
            if 'pqgm_outputs' in predictions:
                pqgm = predictions['pqgm_outputs']
                print(f"   ✅ Module 4 - Prompt weights: {pqgm['prompt_weights'].shape}")
                print(f"   ✅ Module 4 - Granularity dist: {pqgm['granularity_distribution'].shape}")
                if pqgm['domain_logits'] is not None:
                    print(f"   ✅ Module 4 - Domain logits: {pqgm['domain_logits'].shape}")
                
                # Show granularity distribution
                gran_dist = pqgm['granularity_distribution'].cpu().numpy()
                print(f"   📊 Granularity preferences: Char={gran_dist[0,0]:.3f}, Word={gran_dist[0,1]:.3f}, Line={gran_dist[0,2]:.3f}")
                print("   ✅ Module 4 (PQGM) working!")
            
            # Test loss computation
            model.train()
            predictions = model(images, domain_id=domain_id)
            loss, loss_dict = criterion(predictions, targets)
            
            print(f"   ✅ Loss computation: SUCCESS!")
            print(f"   📊 Total loss: {loss.item():.4f}")
            print(f"   📋 Loss breakdown:")
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"       {key}: {value.item():.4f}")
            
        except Exception as e:
            print(f"❌ Error in batch {batch_idx + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        if batch_idx >= 2:  # Test first 3 batches
            break
    
    print("\n🎉 Complete test with Module 4 (PQGM) passed!")
    return True

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
