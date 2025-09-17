import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

class VimTSLoss(nn.Module):
    """
    VimTS Loss Function with Hungarian Matching
    Based on the paper's implementation details
    """
    def __init__(self, 
                 weight_class=2.0,    # αc from paper
                 weight_bbox=5.0,     # L1 loss weight  
                 weight_giou=2.0,     # GIoU loss weight
                 weight_polygon=1.0,  # αp from paper
                 weight_recognition=1.0,  # αr from paper
                 focal_alpha=0.25,    # Focal loss α
                 focal_gamma=2.0):    # Focal loss γ
        super(VimTSLoss, self).__init__()
        
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox  
        self.weight_giou = weight_giou
        self.weight_polygon = weight_polygon
        self.weight_recognition = weight_recognition
        
        # Focal loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Loss functions
        self.bbox_loss_fn = nn.L1Loss(reduction='none')
        self.polygon_loss_fn = nn.L1Loss(reduction='none')
        self.recognition_loss_fn = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict containing:
                - 'pred_logits': [B, N, num_classes] classification logits
                - 'pred_boxes': [B, N, 4] bounding box predictions
                - 'pred_polygons': [B, N, num_points*2] polygon predictions  
                - 'pred_texts': [B, N, max_len, vocab_size] text predictions
            targets: List of target dicts for each image in batch
                
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual losses
        """
        # Step 1: Hungarian matching
        indices = self.hungarian_matching(predictions, targets)
        
        # Step 2: Compute individual losses
        loss_class = self.classification_loss(predictions, targets, indices)
        loss_bbox = self.bbox_loss(predictions, targets, indices)  
        loss_polygon = self.polygon_loss(predictions, targets, indices)
        loss_recognition = self.recognition_loss(predictions, targets, indices)
        
        # Step 3: Combine losses (Equation 3 from paper)
        total_loss = (self.weight_class * loss_class + 
                     self.weight_bbox * loss_bbox +
                     self.weight_giou * 0 +  # GIoU included in bbox_loss
                     self.weight_polygon * loss_polygon +
                     self.weight_recognition * loss_recognition)
        
        loss_dict = {
            'loss_classification': loss_class,
            'loss_bbox': loss_bbox,
            'loss_polygon': loss_polygon, 
            'loss_recognition': loss_recognition,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict
    
    def hungarian_matching(self, predictions, targets):
        """
        Hungarian matching algorithm (Equation 1-2 from paper)
        """
        batch_size = predictions['pred_logits'].shape[0]
        indices = []
        
        for i in range(batch_size):
            pred_logits = predictions['pred_logits'][i]  # [N, num_classes]
            pred_boxes = predictions['pred_boxes'][i]    # [N, 4]
            
            target_classes = targets[i]['labels']        # [M]
            target_boxes = targets[i]['boxes']           # [M, 4]
            
            if len(target_classes) == 0:
                indices.append((torch.tensor([], dtype=torch.long), 
                              torch.tensor([], dtype=torch.long)))
                continue
            
            # Classification cost (Equation 2)
            pred_probs = F.softmax(pred_logits, dim=-1)
            cost_class = -pred_probs[:, target_classes]  # [N, M]
            
            # Bounding box cost (L1 + GIoU)
            cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)  # [N, M]
            cost_giou = -self.generalized_box_iou(pred_boxes, target_boxes)  # [N, M]
            
            # Final cost matrix (Equation 2)
            C = (self.weight_class * cost_class + 
                 self.weight_bbox * cost_bbox + 
                 self.weight_giou * cost_giou)
            
            # Hungarian algorithm
            cost_matrix = C.detach().cpu().numpy()
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            indices.append((torch.tensor(row_indices, dtype=torch.long), 
                          torch.tensor(col_indices, dtype=torch.long)))
            
        return indices
    
    def classification_loss(self, predictions, targets, indices):
        """Focal loss for classification"""
        pred_logits = predictions['pred_logits']  # [B, N, num_classes]
        
        # Create target classes for all predictions
        target_classes = torch.full(pred_logits.shape[:2], 0, 
                                   dtype=torch.long, device=pred_logits.device)
        
        # Set matched predictions to correct class
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[i, src_idx] = targets[i]['labels'][tgt_idx]
        
        # Focal loss computation
        ce_loss = F.cross_entropy(pred_logits.view(-1, pred_logits.shape[-1]), 
                                 target_classes.view(-1), reduction='none')
        ce_loss = ce_loss.view(pred_logits.shape[:2])
        
        p_t = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - p_t) ** self.focal_gamma * ce_loss
        
        return focal_loss.mean()
    
    def bbox_loss(self, predictions, targets, indices):
        """L1 + GIoU loss for bounding boxes"""
        pred_boxes = predictions['pred_boxes']  # [B, N, 4]
        
        # Gather matched predictions and targets
        src_boxes = []
        tgt_boxes = []
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_boxes.append(pred_boxes[i, src_idx])
                tgt_boxes.append(targets[i]['boxes'][tgt_idx])
        
        if len(src_boxes) == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
            
        src_boxes = torch.cat(src_boxes, dim=0)
        tgt_boxes = torch.cat(tgt_boxes, dim=0)
        
        # L1 loss
        l1_loss = self.bbox_loss_fn(src_boxes, tgt_boxes).sum(dim=1)
        
        # GIoU loss  
        giou = self.generalized_box_iou(src_boxes, tgt_boxes)
        giou_loss = 1 - torch.diag(giou)
        
        return (l1_loss + self.weight_giou * giou_loss).mean()
    
    def polygon_loss(self, predictions, targets, indices):
        """L1 loss for polygon coordinates"""
        if 'pred_polygons' not in predictions:
            return torch.tensor(0.0, device=predictions['pred_logits'].device)
            
        pred_polygons = predictions['pred_polygons']  # [B, N, num_points*2]
        
        # Gather matched predictions and targets
        src_polygons = []
        tgt_polygons = []
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0 and 'polygons' in targets[i]:
                src_polygons.append(pred_polygons[i, src_idx])
                tgt_polygons.append(targets[i]['polygons'][tgt_idx])
        
        if len(src_polygons) == 0:
            return torch.tensor(0.0, device=pred_polygons.device)
            
        src_polygons = torch.cat(src_polygons, dim=0)
        tgt_polygons = torch.cat(tgt_polygons, dim=0)
        
        polygon_loss = self.polygon_loss_fn(src_polygons, tgt_polygons).sum(dim=1)
        return polygon_loss.mean()
    
    def recognition_loss(self, predictions, targets, indices):
        """Cross-entropy loss for text recognition"""
        if 'pred_texts' not in predictions:
            return torch.tensor(0.0, device=predictions['pred_logits'].device)
            
        pred_texts = predictions['pred_texts']  # [B, N, max_len, vocab_size]
        
        # Gather matched predictions and targets
        src_texts = []
        tgt_texts = []
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0 and 'texts' in targets[i]:
                src_texts.append(pred_texts[i, src_idx])
                tgt_texts.append(targets[i]['texts'][tgt_idx])
        
        if len(src_texts) == 0:
            return torch.tensor(0.0, device=pred_texts.device)
            
        src_texts = torch.cat(src_texts, dim=0)  # [matched_instances, max_len, vocab_size]
        tgt_texts = torch.cat(tgt_texts, dim=0)  # [matched_instances, max_len]
        
        # Flatten for cross-entropy
        src_texts_flat = src_texts.view(-1, src_texts.shape[-1])
        tgt_texts_flat = tgt_texts.view(-1)
        
        recognition_loss = self.recognition_loss_fn(src_texts_flat, tgt_texts_flat)
        
        # Mask out padding tokens (assuming 0 is padding)
        mask = (tgt_texts_flat != 0).float()
        recognition_loss = (recognition_loss * mask).sum() / mask.sum()
        
        return recognition_loss
    
    def generalized_box_iou(self, boxes1, boxes2):
        """
        Generalized IoU computation
        boxes format: [x1, y1, x2, y2]
        """
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        union = area1[:, None] + area2 - inter
        
        iou = inter / union
        
        # Generalized IoU
        lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        
        whi = (rbi - lti).clamp(min=0)
        areai = whi[:, :, 0] * whi[:, :, 1]
        
        return iou - (areai - union) / areai
