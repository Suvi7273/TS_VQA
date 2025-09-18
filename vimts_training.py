# Simple Training Script for VimTS Base Model
# Train on small dataset and save checkpoint for testing

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# Import your VimTS components
from backbone import VimTSFeatureExtraction
from loss import VimTSLoss

# Basic VimTS Model (same as your test.py)
class MinimalVimTSModel(nn.Module):
    """Minimal VimTS model for testing Modules 1 & 7"""
    def __init__(self, num_classes=2, vocab_size=100, max_text_len=25, num_queries=100):
        super().__init__()
        
        # Module 1: Feature Extraction
        self.feature_extractor = VimTSFeatureExtraction(pretrained=True)
        
        # Minimal query generation for testing
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, 256)
        
        # Prediction heads
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
        
        # Simple query processing
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Global average pooling of features
        pooled_features = features.mean(dim=[2, 3])  # [B, 256]
        
        # Add pooled features to queries
        enhanced_queries = queries + pooled_features.unsqueeze(1)
        
        # Prediction heads
        pred_logits = self.class_head(enhanced_queries)
        pred_boxes = self.bbox_head(enhanced_queries).sigmoid()
        pred_polygons = self.polygon_head(enhanced_queries).sigmoid()
        
        # Text predictions
        text_logits = self.text_head(enhanced_queries)
        pred_texts = text_logits.view(batch_size, self.num_queries, self.max_text_len, self.vocab_size)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'pred_polygons': pred_polygons,
            'pred_texts': pred_texts
        }

# Dataset class (same as your test.py)
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
        print(f" Loaded {len(self.image_ids)} images from {dataset_name}")
    
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
            
            # Polygon from segmentation
            if 'segmentation' in ann and len(ann['segmentation']) > 0:
                poly = np.array(ann['segmentation'][0]).reshape(-1, 2)
                polygon_flat = poly.flatten()[:16]
                if len(polygon_flat) < 16:
                    polygon_flat = np.pad(polygon_flat, (0, 16 - len(polygon_flat)))
            else:
                polygon_flat = np.zeros(16)
            polygons.append(polygon_flat)
            
            # Text tokens
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
        """Convert text to tokens"""
        if isinstance(rec_field, str):
            tokens = [min(ord(c), vocab_size - 1) for c in rec_field[:max_len]]
        elif isinstance(rec_field, list):
            tokens = [min(int(v), vocab_size - 1) for v in rec_field[:max_len]]
        else:
            tokens = []
        
        tokens = [0 if t == 96 else t for t in tokens]
        tokens += [0] * (max_len - len(tokens))
        return tokens[:max_len]

def collate_fn(batch):
    """Collate function for variable image sizes"""
    images, targets = zip(*batch)
    
    # Pad images to same size
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

def train_basic_vimts(dataset_path, num_epochs=20, batch_size=2, learning_rate=1e-4):
    """Train basic VimTS model on small dataset"""
    
    print(" Starting VimTS Basic Training")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    # Create dataset and dataloader
    dataset = VimTSRealDataset(dataset_path, split='train', dataset_name='sample')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Initialize model
    model = MinimalVimTSModel().to(device)
    criterion = VimTSLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f" Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            
            # Forward pass
            predictions = model(images)
            loss, loss_dict = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track loss
            epoch_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{np.mean(epoch_losses):.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
    
    # Save final model
    final_checkpoint_path = "vimts_trained_model.pth"
    save_checkpoint(model, optimizer, num_epochs-1, train_losses[-1], final_checkpoint_path)
    
    print("\n Training completed!")
    print(f" Final model saved to: {final_checkpoint_path}")
    
    return model, train_losses

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': {
            'num_classes': 2,
            'vocab_size': 100,
            'max_text_len': 25,
            'num_queries': 100
        }
    }
    torch.save(checkpoint, filepath)
    print(f" Checkpoint saved: {filepath}")

def test_trained_model(model_path, test_image_path=None):
    """Quick test of trained model"""
    print(f" Testing trained model from: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MinimalVimTSModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f" Model loaded from epoch {checkpoint['epoch']}")
    
    # Test with dummy image if no test image provided
    if test_image_path is None or not os.path.exists(test_image_path):
        print(" Creating dummy test image...")
        dummy_image = torch.randn(1, 3, 640, 480).to(device)
    else:
        print(f" Loading test image: {test_image_path}")
        image = Image.open(test_image_path).convert('RGB')
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        dummy_image = image_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(dummy_image)
    
    print(f" Inference Results:")
    print(f"   Pred logits shape: {predictions['pred_logits'].shape}")
    print(f"   Pred boxes shape: {predictions['pred_boxes'].shape}")
    print(f"   Pred polygons shape: {predictions['pred_polygons'].shape}")
    print(f"   Pred texts shape: {predictions['pred_texts'].shape}")
    
    # Check for confident predictions
    class_probs = torch.softmax(predictions['pred_logits'][0], dim=-1)
    text_scores = class_probs[:, 1]  # Text class scores
    confident_detections = (text_scores > 0.5).sum().item()
    
    print(f"   Confident detections (>0.5): {confident_detections}")
    print(f"   Max confidence: {text_scores.max().item():.3f}")
    
    return predictions

def main():
    """Main function"""
    print(" VimTS Basic Training and Testing")
    print("=" * 50)
    
    # Configuration
    dataset_path = "/content/drive/MyDrive/sample"  # Update this to your dataset path
    num_epochs = 20
    batch_size = 2
    learning_rate = 1e-4
    
    print("Choose option:")
    print("1. Train new model")
    print("2. Test existing model")
    print("3. Train and then test")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Train only
        model, losses = train_basic_vimts(
            dataset_path=dataset_path,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        print(" Training completed!")
        
    elif choice == "2":
        # Test only
        model_path = input("Enter model checkpoint path: ").strip()
        if os.path.exists(model_path):
            test_trained_model(model_path)
        else:
            print(f" Model not found: {model_path}")
            
    elif choice == "3":
        # Train and test
        model, losses = train_basic_vimts(
            dataset_path=dataset_path,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        print("\n" + "="*50)
        print(" Testing trained model...")
        test_trained_model("vimts_trained_model.pth")
        
    else:
        print(" Invalid choice")

if __name__ == "__main__":
    main()
