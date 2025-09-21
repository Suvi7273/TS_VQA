# VimTS Improved Training Strategy
# Address high loss issue with better training approach

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import logging
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import your components
from backbone import VimTSFeatureExtraction
from loss import VimTSLoss
from data_augmentation import create_augmentation_pipeline, TextSpottingAugmentation

class ImprovedVimTSModel(nn.Module):
    """
    Improved VimTS model with better initialization and architecture
    """
    def __init__(self, num_classes=2, vocab_size=100, max_text_len=25, num_queries=100):
        super().__init__()
        
        # Module 1: Feature Extraction
        self.feature_extractor = VimTSFeatureExtraction(pretrained=True)
        
        # Improved query initialization
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, 256)
        
        # Additional feature processing
        self.feature_projection = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )
        
        # Improved prediction heads with proper initialization
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
        
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Proper weight initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
        
        # Special initialization for classification head (bias towards background)
        nn.init.constant_(self.class_head[-1].bias, 0)
        nn.init.constant_(self.class_head[-1].bias[0], -2.0)  # Background bias
        
    def forward(self, images):
        batch_size = images.shape[0]
        
        # Module 1: Feature extraction
        features = self.feature_extractor(images)  # [B, 256, H', W']
        
        # Additional feature processing
        processed_features = self.feature_projection(features)
        
        # Improved query processing
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Multi-scale feature pooling
        avg_pool = F.adaptive_avg_pool2d(processed_features, (1, 1)).flatten(1)  # [B, 256]
        max_pool = F.adaptive_max_pool2d(processed_features, (1, 1)).flatten(1)  # [B, 256] 
        pooled_features = (avg_pool + max_pool) / 2
        
        # Enhance queries with pooled features
        enhanced_queries = queries + pooled_features.unsqueeze(1)
        
        # Apply layer normalization
        enhanced_queries = F.layer_norm(enhanced_queries, [256])
        
        # Predictions
        pred_logits = self.class_head(enhanced_queries)
        pred_boxes = torch.sigmoid(self.bbox_head(enhanced_queries))
        pred_polygons = torch.sigmoid(self.polygon_head(enhanced_queries))
        
        # Text predictions
        text_logits = self.text_head(enhanced_queries)
        pred_texts = text_logits.view(batch_size, self.num_queries, self.max_text_len, self.vocab_size)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'pred_polygons': pred_polygons,
            'pred_texts': pred_texts
        }

class ImprovedVimTSDataset(Dataset):
    """Dataset with augmentation support"""
    def __init__(self, dataset_path, split='train', dataset_name='sample', use_augmentation=True):
        self.dataset_path = dataset_path
        self.split = split
        self.dataset_name = dataset_name
        self.use_augmentation = use_augmentation and (split == 'train')
        
        # Load dataset
        annotation_file = os.path.join(dataset_path, dataset_name, f'{split}.json')
        image_dir = os.path.join(dataset_path, dataset_name, 'img')
        
        with open(annotation_file, 'r') as f:
            coco = json.load(f)
        
        self.images = {img['id']: img for img in coco['images']}
        self.annotations = coco['annotations']
        
        # Group annotations by image_id
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann['image_id'], []).append(ann)
        
        self.image_ids = list(self.images.keys())
        
        # Setup augmentation
        if self.use_augmentation:
            self.augmentation = TextSpottingAugmentation(
                image_size=(640, 640),
                augment_prob=0.9,  # High augmentation probability for small dataset
                strong_augment_prob=0.4
            )
        
        print(f" {split} dataset: {len(self.image_ids)} images, augmentation: {self.use_augmentation}")
        
        # Data repetition for small datasets
        if split == 'train' and len(self.image_ids) < 100:
            self.repeat_factor = max(1, 100 // len(self.image_ids))
            print(f" Small dataset detected, repeating {self.repeat_factor}x for better training")
        else:
            self.repeat_factor = 1
    
    def __len__(self):
        return len(self.image_ids) * self.repeat_factor
    
    def __getitem__(self, idx):
        # Handle repetition
        actual_idx = idx % len(self.image_ids)
        
        image_id = self.image_ids[actual_idx]
        img_info = self.images[image_id]
        ann_list = self.image_to_anns.get(image_id, [])
        
        # Load image
        image_path = os.path.join(self.dataset_path, self.dataset_name, 'img', img_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Parse annotations
        labels, boxes, polygons, texts = [], [], [], []
        
        for ann in ann_list:
            labels.append(ann.get('category_id', 1))
            
            # Bounding box [x, y, w, h] → [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Polygon
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
        
        # Create targets
        targets = {
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.tensor([1], dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float) if boxes else torch.tensor([[10, 10, 50, 50]], dtype=torch.float),
            'polygons': torch.tensor(polygons, dtype=torch.float) if polygons else torch.tensor([[10, 10, 50, 10, 50, 50, 10, 50, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float),
            'texts': torch.tensor(texts, dtype=torch.long) if texts else torch.tensor([[0] * 25], dtype=torch.long)
        }
        
        # Apply augmentation
        if self.use_augmentation:
            try:
                image_tensor, targets = self.augmentation.augment_sample(image, targets)
                return image_tensor, targets
            except Exception as e:
                print(f" Augmentation failed: {e}")
                # Fallback to basic preprocessing
                pass
        
        # Basic preprocessing
        image_array = np.array(image)
        image_tensor = torch.tensor(image_array).permute(2, 0, 1).float() / 255.0
        
        # Resize image
        image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False).squeeze(0)
        
        # Scale boxes to match resized image
        original_h, original_w = image_array.shape[:2]
        scale_x = 640 / original_w
        scale_y = 640 / original_h
        
        scaled_boxes = targets['boxes'].clone()
        scaled_boxes[:, [0, 2]] *= scale_x
        scaled_boxes[:, [1, 3]] *= scale_y
        targets['boxes'] = scaled_boxes
        
        return image_tensor, targets
    
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

class ImprovedTrainer:
    """
    Improved trainer with better training strategy
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = ImprovedVimTSModel().to(self.device)
        
        # Better loss function
        self.criterion = VimTSLoss()
        
        # Improved optimizer with proper hyperparameters
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get('max_lr', 1e-3),
            epochs=config.get('num_epochs', 100),
            steps_per_epoch=config.get('steps_per_epoch', 10),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Training tracking
        self.train_losses = []
        self.best_loss = float('inf')
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup training logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"training_logs_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.log_dir = log_dir
        
    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]
            
            # Forward pass
            predictions = self.model(images)
            
            # Compute loss
            try:
                loss, loss_dict = self.criterion(predictions, targets)
            except Exception as e:
                logging.warning(f"Loss computation failed: {e}")
                continue
            
            # Check for invalid loss
            if not torch.isfinite(loss):
                logging.warning(f"Invalid loss detected: {loss.item()}")
                continue
                
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track loss
            epoch_losses.append(loss.item())
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{np.mean(epoch_losses):.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # Log detailed losses
            if batch_idx % 10 == 0:
                loss_info = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items() if isinstance(v, torch.Tensor)])
                logging.info(f"Batch {batch_idx}: {loss_info}")
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        return avg_loss
    
    def train(self, train_dataset, num_epochs=100, batch_size=2):
        """Main training loop"""
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        logging.info(f" Starting improved training")
        logging.info(f" Dataset size: {len(train_dataset)}")
        logging.info(f" Batch size: {batch_size}")
        logging.info(f" Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(avg_loss)
            
            logging.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(f"best_model.pth", epoch, avg_loss)
                logging.info(f" New best model saved (loss: {avg_loss:.4f})")
            
            # Save regular checkpoints
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", epoch, avg_loss)
            
            # Early stopping if loss is too high
            if avg_loss > 10000:
                logging.warning(f" Very high loss detected: {avg_loss:.2f}")
                logging.info("Consider:")
                logging.info("1. Reducing learning rate")
                logging.info("2. Checking data preprocessing")
                logging.info("3. Adding more regularization")
        
        # Save final model
        self.save_checkpoint("final_model.pth", num_epochs-1, self.train_losses[-1])
        
        # Plot training curve
        self.plot_training_curve()
        
        logging.info("🎉 Training completed!")
        return self.model
    
    def save_checkpoint(self, filename, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'config': self.config
        }
        
        filepath = os.path.join(self.log_dir, filename)
        torch.save(checkpoint, filepath)
    
    def plot_training_curve(self):
        """Plot and save training curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses)
        plt.title('VimTS Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')  # Log scale for better visualization
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.log_dir, 'training_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"📊 Training curve saved: {plot_path}")
    
    @staticmethod
    def collate_fn(batch):
        """Improved collate function"""
        images, targets = zip(*batch)
        
        # Stack images with proper padding
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        
        batch_images = torch.zeros((len(images), 3, max_h, max_w))
        
        for i, img in enumerate(images):
            c, h, w = img.shape
            batch_images[i, :, :h, :w] = img
        
        return batch_images, list(targets)

def main():
    """Main training function with improved strategy"""
    
    # Configuration
    config = {
        'dataset_path': '/content/drive/MyDrive',
        'dataset_name': 'sample',
        'num_epochs': 100,  # More epochs for better convergence
        'batch_size': 2,    # Small batch size for small dataset
        'learning_rate': 5e-5,  # Lower learning rate for stability
        'max_lr': 1e-3,     # Max learning rate for OneCycle
        'weight_decay': 0.01,
        'steps_per_epoch': 10  # Adjust based on dataset size
    }
    
    print(" VimTS Improved Training Strategy")
    print("=" * 50)
    print(" Using data augmentation")
    print(" Better model initialization")
    print(" Improved optimizer settings")
    print(" Learning rate scheduling")
    print(" Gradient clipping")
    print(" Comprehensive logging")
    
    # Create dataset with augmentation
    train_dataset = ImprovedVimTSDataset(
        dataset_path=config['dataset_path'],
        split='train',
        dataset_name=config['dataset_name'],
        use_augmentation=True
    )
    
    # Initialize trainer
    trainer = ImprovedTrainer(config)
    
    # Train model
    trained_model = trainer.train(
        train_dataset=train_dataset,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size']
    )
    
    print(" Expected improvements:")
    print("• Loss should decrease from ~14567 to <500")
    print("• Better convergence with augmentation")
    print("• More stable training with proper initialization")
    print("• Detailed logging for monitoring")

if __name__ == "__main__":
    main()
