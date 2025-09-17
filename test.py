import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

from backbone import VimTSFeatureExtraction
from loss import VimTSLoss

class MinimalVimTSModel(nn.Module):
    """Minimal VimTS model for testing Modules 1 & 7"""
    def __init__(self, num_classes=2, vocab_size=100, max_text_len=25, num_queries=100):
        super().__init__()
        
        # Your Module 1: Feature Extraction
        self.feature_extractor = VimTSFeatureExtraction(pretrained=True)
        
        # Minimal query generation for testing
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, 256)
        
        # Minimal prediction heads
        self.class_head = nn.Linear(256, num_classes + 1)  # +1 for background
        self.bbox_head = nn.Linear(256, 4)
        self.polygon_head = nn.Linear(256, 16)  # 8 points * 2 coords
        self.text_head = nn.Linear(256, max_text_len * vocab_size)
        
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        
    def forward(self, images):
        batch_size = images.shape[0]
        
        # Module 1: Feature extraction
        features = self.feature_extractor(images)  # [B, 256, H', W']
        
        # Simple query processing for testing
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_queries, 256]
        
        # Global average pooling of features for simplicity
        pooled_features = features.mean(dim=[2, 3])  # [B, 256]
        
        # Add pooled features to queries (simplified attention)
        enhanced_queries = queries + pooled_features.unsqueeze(1)  # [B, num_queries, 256]
        
        # Prediction heads
        pred_logits = self.class_head(enhanced_queries)  # [B, num_queries, num_classes+1]
        pred_boxes = self.bbox_head(enhanced_queries).sigmoid() * 640  # [B, num_queries, 4]
        pred_polygons = self.polygon_head(enhanced_queries).sigmoid() * 640  # [B, num_queries, 16]
        
        # Text predictions
        text_logits = self.text_head(enhanced_queries)  # [B, num_queries, max_len*vocab]
        pred_texts = text_logits.view(batch_size, self.num_queries, self.max_text_len, self.vocab_size)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'pred_polygons': pred_polygons,
            'pred_texts': pred_texts
        }

import os
import json
import cv2
from PIL import Image
from torch.utils.data import Dataset

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


# Usage for real training
def create_real_dataloader(dataset_path):
    """Create DataLoader with real VimTS datasets"""
    
    # Dataset paths structure:
    # dataset_path/
    # ├── totaltext/
    # │   ├── train_images/
    # │   ├── test_images/
    # │   ├── train.json
    # │   └── test.json
    # ├── CTW1500/
    # │   ├── ctwtrain_text_image/
    # │   ├── ctwtest_text_image/
    # │   └── annotations/
    # └── icdar2015/...
    
    train_dataset = VimTSRealDataset(
        dataset_path=dataset_path,
        split='train',
        dataset_name='sample'  # or 'ctw1500', 'icdar2015'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # Adjust based on GPU memory
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return train_loader


def dry_run_test(dataset_path):
    """Main dry run testing function"""
    
    print(" Starting VimTS Dry Run Test with REAL dataset...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MinimalVimTSModel().to(device)
    criterion = VimTSLoss()

    # dataset_path = r"G:\sample"
    # Use real dataloader
    dataloader = create_real_dataloader(dataset_path)

    model.eval()
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        
        try:
            # Forward pass
            with torch.no_grad():
                predictions = model(images)
            
            print(f"✅ Batch {batch_idx + 1}:")
            print(f"   Images shape: {images.shape}")
            print(f"   Pred logits shape: {predictions['pred_logits'].shape}")
            print(f"   Pred boxes shape: {predictions['pred_boxes'].shape}")
            print(f"   Pred polygons shape: {predictions['pred_polygons'].shape}")
            print(f"   Pred texts shape: {predictions['pred_texts'].shape}")
            
            # Test loss computation
            model.train()
            predictions = model(images)
            loss, loss_dict = criterion(predictions, targets)
            
            print(f"   Loss computation: ✅")
            print(f"   Total loss: {loss.item():.4f}")
            print(f"   Loss breakdown: {loss_dict}")
            
        except Exception as e:
            print(f"❌ Error in batch {batch_idx + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    
    print("🎉 Dry run completed successfully!")
    return True

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


# Test gradient flow
def test_gradient_flow():
    """Test if gradients flow properly through the model"""
    print("🔍 Testing gradient flow...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MinimalVimTSModel().to(device)
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
        # Forward pass
        predictions = model(images)
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

if __name__ == "__main__":
    dataset_path = r"/content/drive/MyDrive"  # update this
    success = dry_run_test(dataset_path)
    if success:
        success = test_gradient_flow()
    
    if success:
        print("\n ALL TESTS PASSED! Your Modules 1 & 7 are working correctly.")
        print(" Ready to implement Module 2: Query Initialization")
    else:
        print("\n Tests failed. Please fix the issues before proceeding.")

