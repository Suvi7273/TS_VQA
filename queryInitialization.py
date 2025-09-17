import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import VimTSFeatureExtraction

class QueryInitialization(nn.Module):
    """
    Module 2: Query Initialization
    Generates detection and recognition queries from enhanced features
    """
    def __init__(self, 
                 feature_dim=256,
                 num_detection_queries=100,
                 num_recognition_queries=25,
                 hidden_dim=256):
        super(QueryInitialization, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_detection_queries = num_detection_queries
        self.num_recognition_queries = num_recognition_queries
        self.hidden_dim = hidden_dim
        
        # Coarse prediction heads for query selection
        self.coarse_class_head = nn.Linear(feature_dim, 2)  # text/no-text
        self.coarse_bbox_head = nn.Linear(feature_dim, 4)   # coarse bounding boxes
        
        # Query generation networks
        self.detection_query_generator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.recognition_query_generator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Learnable query embeddings
        self.detection_queries = nn.Embedding(num_detection_queries, feature_dim)
        self.recognition_queries = nn.Embedding(num_recognition_queries, feature_dim)
        
        # Position embeddings
        self.pos_embed_2d = nn.Parameter(torch.zeros(1, feature_dim, 50, 50))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, enhanced_features):
        """
        Args:
            enhanced_features: [B, C, H, W] from Module 1
            
        Returns:
            detection_queries: [B, num_detection_queries, C]
            recognition_queries: [B, num_recognition_queries, C]
            coarse_predictions: dict with coarse classification and bbox predictions
        """
        batch_size, C, H, W = enhanced_features.shape
        
        # Add positional encoding
        pos_embed = F.interpolate(
            self.pos_embed_2d, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        enhanced_features = enhanced_features + pos_embed
        
        # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        features_flat = enhanced_features.flatten(2).permute(0, 2, 1)
        
        # Coarse predictions for query selection
        coarse_class_logits = self.coarse_class_head(features_flat)  # [B, H*W, 2]
        coarse_bbox_pred = self.coarse_bbox_head(features_flat)      # [B, H*W, 4]
        
        # Select top-N locations for query initialization
        detection_queries = self._generate_detection_queries(
            features_flat, coarse_class_logits, coarse_bbox_pred
        )
        
        recognition_queries = self._generate_recognition_queries(
            features_flat, coarse_class_logits
        )
        
        coarse_predictions = {
            'coarse_class_logits': coarse_class_logits,
            'coarse_bbox_pred': coarse_bbox_pred
        }
        
        return detection_queries, recognition_queries, coarse_predictions
    
    def _generate_detection_queries(self, features, class_logits, bbox_pred):
        """Generate detection queries based on coarse predictions"""
        batch_size = features.shape
        
        # Get confidence scores for text regions
        text_confidence = F.softmax(class_logits, dim=-1)[..., 1]  # [B, H*W]
        
        # Select top-N confident locations per batch
        detection_queries_list = []
        
        for b in range(batch_size):
            confidence_b = text_confidence[b]  # [H*W]
            features_b = features[b]  # [H*W, C]
            
            # Select top-N locations
            if confidence_b.numel() >= self.num_detection_queries:
                _, top_indices = torch.topk(
                    confidence_b, 
                    self.num_detection_queries, 
                    largest=True
                )
            else:
                # If not enough locations, pad with random selections
                top_indices = torch.randperm(confidence_b.numel())[:self.num_detection_queries]
                if len(top_indices) < self.num_detection_queries:
                    padding_indices = torch.randint(
                        0, confidence_b.numel(), 
                        (self.num_detection_queries - len(top_indices),)
                    )
                    top_indices = torch.cat([top_indices, padding_indices])
            
            # Extract features for selected locations
            selected_features = features_b[top_indices]  # [num_detection_queries, C]
            
            # Generate queries using the query generator network
            detection_queries_b = self.detection_query_generator(selected_features)
            
            # Add learnable embeddings
            learnable_queries = self.detection_queries.weight  # [num_detection_queries, C]
            detection_queries_b = detection_queries_b + learnable_queries
            
            detection_queries_list.append(detection_queries_b)
        
        detection_queries = torch.stack(detection_queries_list, dim=0)  # [B, N_det, C]
        return detection_queries
    
    def _generate_recognition_queries(self, features, class_logits):
        """Generate recognition queries for character/word recognition"""
        batch_size = features.shape
        
        # Similar to detection queries but for recognition
        text_confidence = F.softmax(class_logits, dim=-1)[..., 1]  # [B, H*W]
        
        recognition_queries_list = []
        
        for b in range(batch_size):
            confidence_b = text_confidence[b]
            features_b = features[b]
            
            # Select top locations for recognition
            if confidence_b.numel() >= self.num_recognition_queries:
                _, top_indices = torch.topk(
                    confidence_b, 
                    self.num_recognition_queries, 
                    largest=True
                )
            else:
                top_indices = torch.randperm(confidence_b.numel())[:self.num_recognition_queries]
                if len(top_indices) < self.num_recognition_queries:
                    padding_indices = torch.randint(
                        0, confidence_b.numel(), 
                        (self.num_recognition_queries - len(top_indices),)
                    )
                    top_indices = torch.cat([top_indices, padding_indices])
            
            selected_features = features_b[top_indices]
            
            # Generate recognition queries
            recognition_queries_b = self.recognition_query_generator(selected_features)
            
            # Add learnable embeddings
            learnable_queries = self.recognition_queries.weight
            recognition_queries_b = recognition_queries_b + learnable_queries
            
            recognition_queries_list.append(recognition_queries_b)
        
        recognition_queries = torch.stack(recognition_queries_list, dim=0)  # [B, N_rec, C]
        return recognition_queries

# Integration with your existing model
class VimTSWithQueryInit(nn.Module):
    """Updated VimTS model with Module 2"""
    def __init__(self, num_classes=2, vocab_size=100, max_text_len=25):
        super().__init__()
        
        # Module 1: Feature Extraction (your existing implementation)
        self.feature_extractor = VimTSFeatureExtraction(pretrained=True)
        
        # Module 2: Query Initialization  
        self.query_initializer = QueryInitialization(
            feature_dim=256,
            num_detection_queries=100,
            num_recognition_queries=25
        )
        
        # Prediction heads (simplified for now)
        self.class_head = nn.Linear(256, num_classes + 1)
        self.bbox_head = nn.Linear(256, 4)
        self.polygon_head = nn.Linear(256, 16)
        self.text_head = nn.Linear(256, max_text_len * vocab_size)
        
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
    
    def forward(self, images):
        batch_size = images.shape[0]
        
        # Module 1: Feature extraction
        enhanced_features = self.feature_extractor(images)
        
        # Module 2: Query initialization
        detection_queries, recognition_queries, coarse_preds = self.query_initializer(enhanced_features)
        
        # Combine queries for unified processing
        all_queries = torch.cat([detection_queries, recognition_queries], dim=1)  # [B, N_total, C]
        
        # Prediction heads
        pred_logits = self.class_head(all_queries)
        pred_boxes = self.bbox_head(all_queries).sigmoid() * 640
        pred_polygons = self.polygon_head(all_queries).sigmoid() * 640
        
        text_logits = self.text_head(all_queries)
        pred_texts = text_logits.view(batch_size, -1, self.max_text_len, self.vocab_size)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'pred_polygons': pred_polygons,
            'pred_texts': pred_texts,
            'coarse_predictions': coarse_preds
        }
