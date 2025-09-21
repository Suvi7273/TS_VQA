# VimTS Advanced Data Augmentation for Text Spotting
# Specialized augmentations for text detection and recognition

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import math

class TextSpottingAugmentation:
    """
    Advanced data augmentation specifically designed for text spotting
    Based on VimTS paper recommendations and text spotting best practices
    """
    
    def __init__(self, 
                 image_size=(640, 640),
                 augment_prob=0.8,
                 strong_augment_prob=0.3):
        self.image_size = image_size
        self.augment_prob = augment_prob
        self.strong_augment_prob = strong_augment_prob
        
        # Define augmentation pipeline
        self.geometric_transforms = A.Compose([
            # Rotation (small angles to preserve text readability)
            A.Rotate(limit=15, p=0.5),
            
            # Perspective and affine transforms
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2, 
                rotate_limit=10,
                p=0.5
            ),
            
            # Elastic transform (subtle)
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                alpha_affine=5,
                p=0.2
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        ))
        
        self.photometric_transforms = A.Compose([
            # Color and lighting variations
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3, 
                saturation=0.3,
                hue=0.1,
                p=0.6
            ),
            
            # Blur and noise (careful with text)
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=2, p=0.2)
            ], p=0.3),
            
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.4),
                A.ISONoise(p=0.3),
            ], p=0.3),
            
            # Weather and environmental effects
            A.OneOf([
                A.RandomShadow(p=0.3),
                A.RandomSunFlare(p=0.2),
                A.RandomFog(p=0.2),
            ], p=0.2),
        ])
        
        self.strong_transforms = A.Compose([
            # More aggressive transforms for robustness
            A.RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.6
            ),
            
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            
            # Distortions
            A.OpticalDistortion(
                distort_limit=0.1,
                shift_limit=0.1,
                p=0.3
            ),
            
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                p=0.3
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        ))
        
        # Final resize and normalization
        self.final_transform = A.Compose([
            A.Resize(image_size[0], image_size[1], p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        ))
        
    def augment_sample(self, image, targets):
        """
        Apply augmentation to image and corresponding annotations
        
        Args:
            image: PIL Image or numpy array
            targets: dict with 'boxes', 'labels', 'polygons', 'texts'
        
        Returns:
            augmented_image: torch.Tensor
            augmented_targets: dict with updated annotations
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Prepare bounding boxes for albumentations
        boxes = targets['boxes'].cpu().numpy() if torch.is_tensor(targets['boxes']) else targets['boxes']
        labels = targets['labels'].cpu().numpy() if torch.is_tensor(targets['labels']) else targets['labels']
        
        # Convert boxes to pascal_voc format if needed
        if boxes.shape[1] == 4:
            # Assume format is [x1, y1, x2, y2] or [x, y, w, h]
            if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
                # Convert [x, y, w, h] to [x1, y1, x2, y2]
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2] 
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        try:
            # Apply geometric transforms
            if random.random() < self.augment_prob:
                transformed = self.geometric_transforms(
                    image=image,
                    bboxes=boxes,
                    class_labels=labels
                )
                image = transformed['image']
                boxes = np.array(transformed['bboxes'])
                labels = np.array(transformed['class_labels'])
            
            # Apply photometric transforms
            if random.random() < self.augment_prob:
                transformed = self.photometric_transforms(image=image)
                image = transformed['image']
            
            # Apply strong transforms occasionally
            if random.random() < self.strong_augment_prob:
                transformed = self.strong_transforms(
                    image=image,
                    bboxes=boxes,
                    class_labels=labels
                )
                image = transformed['image']
                boxes = np.array(transformed['bboxes'])
                labels = np.array(transformed['class_labels'])
            
            # Final resize and normalization
            transformed = self.final_transform(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            
            final_image = transformed['image']
            final_boxes = np.array(transformed['bboxes'])
            final_labels = np.array(transformed['class_labels'])
            
        except Exception as e:
            print(f" Augmentation failed, using original: {str(e)}")
            # Fallback to basic transform
            basic_transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            transformed = basic_transform(image=image)
            final_image = transformed['image']
            final_boxes = boxes
            final_labels = labels
        
        # Update targets
        augmented_targets = targets.copy()
        augmented_targets['boxes'] = torch.tensor(final_boxes, dtype=torch.float)
        augmented_targets['labels'] = torch.tensor(final_labels, dtype=torch.long)
        
        # Handle polygons (simple scaling)
        if 'polygons' in targets:
            original_h, original_w = image.shape[:2] if len(image.shape) == 3 else (image.shape[0], image.shape[1])
            scale_x = self.image_size[1] / original_w
            scale_y = self.image_size[0] / original_h
            
            polygons = targets['polygons'].cpu().numpy() if torch.is_tensor(targets['polygons']) else targets['polygons']
            scaled_polygons = polygons.copy()
            scaled_polygons[:, 0::2] *= scale_x  # x coordinates
            scaled_polygons[:, 1::2] *= scale_y  # y coordinates
            
            augmented_targets['polygons'] = torch.tensor(scaled_polygons, dtype=torch.float)
        
        # Keep text labels unchanged
        if 'texts' in targets:
            augmented_targets['texts'] = targets['texts']
        
        return final_image, augmented_targets

class MixUpAugmentation:
    """
    MixUp augmentation for text spotting - mixes two images and their annotations
    """
    
    def __init__(self, alpha=0.2, prob=0.3):
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, image1, targets1, image2, targets2):
        """Apply MixUp augmentation"""
        
        if random.random() > self.prob:
            return image1, targets1
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2
        
        # Mix targets (combine both sets of annotations)
        mixed_targets = {}
        
        # Concatenate labels
        labels1 = targets1['labels']
        labels2 = targets2['labels'] 
        mixed_targets['labels'] = torch.cat([labels1, labels2])
        
        # Concatenate boxes
        boxes1 = targets1['boxes']
        boxes2 = targets2['boxes']
        mixed_targets['boxes'] = torch.cat([boxes1, boxes2])
        
        # Concatenate polygons
        if 'polygons' in targets1 and 'polygons' in targets2:
            polygons1 = targets1['polygons']
            polygons2 = targets2['polygons']
            mixed_targets['polygons'] = torch.cat([polygons1, polygons2])
        
        # Concatenate texts
        if 'texts' in targets1 and 'texts' in targets2:
            texts1 = targets1['texts']
            texts2 = targets2['texts']
            mixed_targets['texts'] = torch.cat([texts1, texts2])
        
        return mixed_image, mixed_targets

class CutMixAugmentation:
    """
    CutMix augmentation for text spotting
    """
    
    def __init__(self, alpha=1.0, prob=0.3):
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, image1, targets1, image2, targets2):
        """Apply CutMix augmentation"""
        
        if random.random() > self.prob:
            return image1, targets1
            
        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get image dimensions
        _, h, w = image1.shape
        
        # Sample cut area
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Random position
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Compute cut box
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h) 
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cut
        mixed_image = image1.clone()
        mixed_image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
        
        # Filter annotations based on cut area
        boxes1 = targets1['boxes']
        boxes2 = targets2['boxes']
        
        # Keep annotations from image1 that don't overlap much with cut area
        keep_idx1 = []
        for i, box in enumerate(boxes1):
            box_x1, box_y1, box_x2, box_y2 = box
            # Check overlap with cut area
            overlap_area = max(0, min(box_x2, x2) - max(box_x1, x1)) * max(0, min(box_y2, y2) - max(box_y1, y1))
            box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
            if overlap_area / box_area < 0.5:  # Keep if less than 50% overlap
                keep_idx1.append(i)
        
        # Keep annotations from image2 that are within cut area
        keep_idx2 = []
        for i, box in enumerate(boxes2):
            box_x1, box_y1, box_x2, box_y2 = box
            # Check if box center is in cut area
            box_cx = (box_x1 + box_x2) / 2
            box_cy = (box_y1 + box_y2) / 2
            if x1 <= box_cx <= x2 and y1 <= box_cy <= y2:
                keep_idx2.append(i)
        
        # Combine annotations
        mixed_targets = {}
        
        if keep_idx1:
            keep_boxes1 = boxes1[keep_idx1]
            keep_labels1 = targets1['labels'][keep_idx1]
        else:
            keep_boxes1 = torch.empty((0, 4))
            keep_labels1 = torch.empty((0,), dtype=torch.long)
            
        if keep_idx2:
            keep_boxes2 = boxes2[keep_idx2]
            keep_labels2 = targets2['labels'][keep_idx2]
        else:
            keep_boxes2 = torch.empty((0, 4))
            keep_labels2 = torch.empty((0,), dtype=torch.long)
        
        mixed_targets['boxes'] = torch.cat([keep_boxes1, keep_boxes2])
        mixed_targets['labels'] = torch.cat([keep_labels1, keep_labels2])
        
        # Handle other targets similarly
        if 'polygons' in targets1 and 'polygons' in targets2:
            keep_poly1 = targets1['polygons'][keep_idx1] if keep_idx1 else torch.empty((0, 16))
            keep_poly2 = targets2['polygons'][keep_idx2] if keep_idx2 else torch.empty((0, 16))
            mixed_targets['polygons'] = torch.cat([keep_poly1, keep_poly2])
            
        if 'texts' in targets1 and 'texts' in targets2:
            keep_text1 = targets1['texts'][keep_idx1] if keep_idx1 else torch.empty((0, 25), dtype=torch.long)
            keep_text2 = targets2['texts'][keep_idx2] if keep_idx2 else torch.empty((0, 25), dtype=torch.long)
            mixed_targets['texts'] = torch.cat([keep_text1, keep_text2])
        
        return mixed_image, mixed_targets

# Augmentation pipeline integration
def create_augmentation_pipeline(config):
    """
    Create complete augmentation pipeline for VimTS training
    
    Returns:
        train_augmentation: Training augmentation pipeline
        val_augmentation: Validation augmentation pipeline
    """
    
    # Training augmentation (aggressive)
    train_aug = TextSpottingAugmentation(
        image_size=config.get('image_size', (640, 640)),
        augment_prob=config.get('augment_prob', 0.8),
        strong_augment_prob=config.get('strong_augment_prob', 0.3)
    )
    
    # Validation augmentation (minimal)
    val_aug = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Additional augmentations
    mixup_aug = MixUpAugmentation(alpha=0.2, prob=0.3)
    cutmix_aug = CutMixAugmentation(alpha=1.0, prob=0.3)
    
    return {
        'train_augmentation': train_aug,
        'val_augmentation': val_aug,
        'mixup_augmentation': mixup_aug,
        'cutmix_augmentation': cutmix_aug
    }

# Usage example in dataset
class AugmentedVimTSDataset:
    """Example of how to integrate augmentations into dataset"""
    
    def __init__(self, dataset_path, split='train', augmentation_config=None):
        # ... (your existing dataset code)
        
        if split == 'train' and augmentation_config:
            self.augmentations = create_augmentation_pipeline(augmentation_config)
            self.use_augmentation = True
        else:
            self.use_augmentation = False
    
    def __getitem__(self, idx):
        # ... (load image and targets as before)
        
        if self.use_augmentation:
            # Apply augmentation
            augmented_image, augmented_targets = self.augmentations['train_augmentation'].augment_sample(
                image, targets
            )
            return augmented_image, augmented_targets
        else:
            # Basic preprocessing for validation
            basic_transform = A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            transformed = basic_transform(image=np.array(image))
            return transformed['image'], targets
