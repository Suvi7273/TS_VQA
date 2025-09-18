# VimTS Domain-Specific Adapter Training Script
# Parameter-Efficient Fine-tuning for Cross-Domain Text Spotting

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import json
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import yaml
from PIL import Image

# Import VimTS components
from loss import VimTSLoss

# Set up logging
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"adapter_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

# VimTS Complete Model (same as before)
class VimTSWithTaskAdapter(nn.Module):
    """Complete VimTS Model with Task-Aware Adapter"""
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
        
        # Module 4: PQGM
        from pqgm import PQGM
        self.pqgm = PQGM(
            d_model=256,
            num_heads=8,
            num_domains=num_domains,
            max_prompt_len=50,
            dropout=0.1
        )
        
        # Module 5: Task-Aware Adapter
        from taa import TaskAwareAdapter
        self.task_adapter = TaskAwareAdapter(
            d_model=256,
            num_tasks=num_tasks,
            adapter_dim=64,
            use_lora=True,
            use_multi_task=True
        )
        
        # Legacy heads for compatibility
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
        """Complete forward pass"""
        batch_size = images.shape[0]
        
        # Module 1: Feature Extraction
        enhanced_features = self.feature_extractor(images)
        
        # Prepare visual features for decoder and PQGM
        B, C, H, W = enhanced_features.shape
        visual_features = enhanced_features.flatten(2).permute(0, 2, 1)
        
        # Module 2: Query Initialization
        detection_queries, recognition_queries, coarse_preds = self.query_initializer(enhanced_features)
        all_queries = torch.cat([detection_queries, recognition_queries], dim=1)
        
        # Module 3: Decoder
        decoder_queries, attention_weights = self.decoder(all_queries, visual_features)
        
        # Module 4: PQGM
        pqgm_queries, pqgm_outputs = self.pqgm(
            decoder_queries, 
            visual_features, 
            domain_id=domain_id,
            granularity_hints=granularity_hints,
            training=self.training
        )
        
        # Module 5: Task-Aware Adapter
        if self.use_adapter:
            adapted_queries, adapter_outputs = self.task_adapter(pqgm_queries, task_id=task_id)
            predictions = self.task_adapter.get_task_specific_predictions(
                adapted_queries, self.max_text_len, self.vocab_size
            )
        else:
            adapted_queries = pqgm_queries
            adapter_outputs = {'adapter_enabled': False}
            
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
        
        return {
            **predictions,
            'coarse_predictions': coarse_preds,
            'attention_weights': attention_weights,
            'pqgm_outputs': pqgm_outputs,
            'adapter_outputs': adapter_outputs
        }
    
    def enable_adapter_training(self, freeze_backbone=True):
        """Enable parameter-efficient adapter training"""
        self.use_adapter = True
        self.task_adapter.enable_adapter_mode()
        
        if freeze_backbone:
            # Freeze backbone modules for efficient training
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.query_initializer.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            # Keep PQGM trainable for domain adaptation
        
        logging.info("✅ Adapter training mode enabled")
    
    def get_adapter_parameters(self):
        """Get only adapter parameters for efficient training"""
        adapter_params = []
        if self.use_adapter:
            adapter_params.extend(list(self.task_adapter.parameters()))
            adapter_params.extend(list(self.pqgm.parameters()))
        return [p for p in adapter_params if p.requires_grad]
    
    def get_parameter_statistics(self):
        """Get parameter count statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adapter_params = sum(p.numel() for p in self.get_adapter_parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'adapter_parameters': adapter_params,
            'training_efficiency': (1 - trainable_params / total_params) * 100,
            'adapter_percentage': (adapter_params / total_params) * 100
        }

# Dataset Class (same as before but with domain labeling)
class VimTSDomainDataset(Dataset):
    """Domain-specific dataset for adapter training"""
    def __init__(self, dataset_path, split='train', dataset_name='sample', domain_id=0):
        self.dataset_path = dataset_path
        self.split = split
        self.dataset_name = dataset_name
        self.domain_id = domain_id
        
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
        logging.info(f"📊 Domain {domain_id} dataset: {len(self.image_ids)} images")
    
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
            'texts': torch.tensor(texts, dtype=torch.long),
            'domain_id': torch.tensor(self.domain_id, dtype=torch.long)  # Domain label
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

def create_domain_dataloaders(config):
    """Create dataloaders for domain-specific training"""
    dataloaders = {}
    
    for domain_name, domain_config in config['domains'].items():
        # Training dataset
        train_dataset = VimTSDomainDataset(
            dataset_path=domain_config['dataset_path'],
            split='train',
            dataset_name=domain_config['dataset_name'],
            domain_id=domain_config['domain_id']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config['training']['num_workers']
        )
        
        # Validation dataset (optional)
        if domain_config.get('has_validation', False):
            val_dataset = VimTSDomainDataset(
                dataset_path=domain_config['dataset_path'],
                split='val',
                dataset_name=domain_config['dataset_name'],
                domain_id=domain_config['domain_id']
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=config['training']['num_workers']
            )
        else:
            val_loader = None
        
        dataloaders[domain_name] = {
            'train': train_loader,
            'val': val_loader,
            'domain_id': domain_config['domain_id']
        }
    
    return dataloaders

class AdapterTrainer:
    """VimTS Adapter Trainer for Domain-Specific Fine-tuning"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f" Using device: {self.device}")
        
        # Initialize model
        self.model = VimTSWithTaskAdapter(
            num_classes=config['model']['num_classes'],
            vocab_size=config['model']['vocab_size'],
            max_text_len=config['model']['max_text_len'],
            num_domains=config['model']['num_domains'],
            num_tasks=config['model']['num_tasks'],
            use_adapter=True
        ).to(self.device)
        
        # Enable adapter training
        self.model.enable_adapter_training(freeze_backbone=config['training']['freeze_backbone'])
        
        # Print parameter statistics
        stats = self.model.get_parameter_statistics()
        logging.info(f"   Parameter Statistics:")
        logging.info(f"   Total parameters: {stats['total_parameters']:,}")
        logging.info(f"   Trainable parameters: {stats['trainable_parameters']:,}")
        logging.info(f"   Training efficiency: {stats['training_efficiency']:.1f}% reduction")
        
        # Initialize optimizer for adapter parameters only
        adapter_params = self.model.get_adapter_parameters()
        self.optimizer = torch.optim.AdamW(
            adapter_params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['learning_rate'] * 0.01
        )
        
        # Loss function
        self.criterion = VimTSLoss()
        
        # Training metrics
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_domain(self, domain_name, dataloader_dict, num_epochs):
        """Train adapter for specific domain"""
        train_loader = dataloader_dict['train']
        val_loader = dataloader_dict['val']
        domain_id = dataloader_dict['domain_id']
        
        logging.info(f"🎯 Training adapter for domain: {domain_name} (ID: {domain_id})")
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, domain_id, epoch)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, domain_id, epoch)
                self.val_losses.append(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_adapter_checkpoint(domain_name, epoch, is_best=True)
            
            self.train_losses.append(train_loss)
            self.scheduler.step()
            
            # Log epoch results
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
            if val_loader is not None:
                logging.info(f"Validation Loss: {val_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_adapter_checkpoint(domain_name, epoch)
        
        # Save final checkpoint
        self.save_adapter_checkpoint(domain_name, num_epochs-1, is_final=True)
        
    def _train_epoch(self, train_loader, domain_id, epoch):
        """Training for one epoch"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]
            
            # Forward pass with domain_id and task_id
            task_id = domain_id  # Use domain_id as task_id for simplicity
            predictions = self.model(images, domain_id=domain_id, task_id=task_id)
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.get_adapter_parameters(), 
                max_norm=self.config['training']['max_grad_norm']
            )
            
            self.optimizer.step()
            
            # Track loss
            epoch_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{np.mean(epoch_losses):.4f}'
            })
            
            # Log detailed losses periodically
            if batch_idx % self.config['training']['log_every'] == 0:
                logging.info(f"Batch {batch_idx}: {loss_dict}")
        
        return np.mean(epoch_losses)
    
    def _validate_epoch(self, val_loader, domain_id, epoch):
        """Validation for one epoch"""
        self.model.eval()
        epoch_losses = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]
                
                # Forward pass
                task_id = domain_id
                predictions = self.model(images, domain_id=domain_id, task_id=task_id)
                
                # Compute loss
                loss, _ = self.criterion(predictions, targets)
                epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)
    
    def save_adapter_checkpoint(self, domain_name, epoch, is_best=False, is_final=False):
        """Save adapter checkpoint"""
        checkpoint_dir = os.path.join(self.config['training']['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Checkpoint data
        checkpoint = {
            'epoch': epoch,
            'domain_name': domain_name,
            'model_state_dict': self.model.state_dict(),
            'adapter_state_dict': self.model.task_adapter.state_dict(),
            'pqgm_state_dict': self.model.pqgm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Save different types of checkpoints
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, f'{domain_name}_adapter_best.pth')
            logging.info(f" Saving best checkpoint: {checkpoint_path}")
        elif is_final:
            checkpoint_path = os.path.join(checkpoint_dir, f'{domain_name}_adapter_final.pth')
            logging.info(f" Saving final checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'{domain_name}_adapter_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save adapter-only weights (smaller file)
        adapter_only_path = os.path.join(checkpoint_dir, f'{domain_name}_adapter_only.pth')
        torch.save({
            'adapter_state_dict': self.model.task_adapter.state_dict(),
            'pqgm_state_dict': self.model.pqgm.state_dict(),
            'domain_name': domain_name,
            'epoch': epoch
        }, adapter_only_path)
    
    def load_adapter_checkpoint(self, checkpoint_path):
        """Load adapter checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logging.info(f" Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

def train_multi_domain(config):
    """Train adapters for multiple domains"""
    # Setup logging
    log_file = setup_logging(config['training']['output_dir'])
    logging.info(f" Starting VimTS Domain-Specific Adapter Training")
    logging.info(f" Log file: {log_file}")
    
    # Create dataloaders
    dataloaders = create_domain_dataloaders(config)
    logging.info(f" Created dataloaders for {len(dataloaders)} domains")
    
    # Initialize trainer
    trainer = AdapterTrainer(config)
    
    # Train adapters for each domain
    for domain_name, dataloader_dict in dataloaders.items():
        logging.info(f"\n{'='*50}")
        logging.info(f" Training Domain: {domain_name}")
        logging.info(f"{'='*50}")
        
        trainer.train_domain(
            domain_name=domain_name,
            dataloader_dict=dataloader_dict,
            num_epochs=config['training']['num_epochs']
        )
        
        logging.info(f" Completed training for domain: {domain_name}")
    
    logging.info(f"\n Completed training for all domains!")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='VimTS Domain-Specific Adapter Training')
    parser.add_argument('--config', type=str, required=True, 
                      help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Start training
    train_multi_domain(config)

if __name__ == "__main__":
    main()
