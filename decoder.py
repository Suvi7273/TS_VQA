# VimTS Module 3: Decoder Implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class VisionLanguageCommunication(nn.Module):
    """
    Vision-Language Communication Module
    Enables interaction between visual features and text queries
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super(VisionLanguageCommunication, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # Self-attention for queries
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Cross-attention for vision-language interaction
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, visual_features, query_pos=None, feat_pos=None):
        """
        Args:
            queries: [B, N, C] detection/recognition queries
            visual_features: [B, H*W, C] flattened visual features
            query_pos: [B, N, C] positional encoding for queries (optional)
            feat_pos: [B, H*W, C] positional encoding for features (optional)
            
        Returns:
            enhanced_queries: [B, N, C] enhanced queries after interaction
        """
        # Add positional encoding if provided
        q = queries + query_pos if query_pos is not None else queries
        k = v = visual_features + feat_pos if feat_pos is not None else visual_features
        
        # Self-attention among queries
        queries2, _ = self.self_attn(q, q, q)
        queries = queries + self.dropout(queries2)
        queries = self.norm1(queries)
        
        # Cross-attention with visual features
        queries2, attn_weights = self.cross_attn(queries, k, v)
        queries = queries + self.dropout(queries2)
        queries = self.norm2(queries)
        
        # Feed-forward network
        queries2 = self.ffn(queries)
        queries = queries + self.dropout(queries2)
        queries = self.norm3(queries)
        
        return queries, attn_weights

class TransformerDecoderLayer(nn.Module):
    """
    Custom Transformer Decoder Layer for VimTS
    """
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        self.vision_lang_comm = VisionLanguageCommunication(d_model, nhead, dropout)
        
        # Additional processing layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        
    def forward(self, tgt, memory, tgt_pos=None, memory_pos=None):
        """
        Args:
            tgt: [B, N, C] target queries (detection/recognition)
            memory: [B, H*W, C] visual memory features
            tgt_pos: [B, N, C] positional encoding for queries
            memory_pos: [B, H*W, C] positional encoding for features
            
        Returns:
            enhanced_queries: [B, N, C]
            attention_weights: attention maps
        """
        # Vision-language communication
        tgt2, attn_weights = self.vision_lang_comm(tgt, memory, tgt_pos, memory_pos)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Feed-forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        return tgt, attn_weights

class VimTSDecoder(nn.Module):
    """
    Module 3: VimTS Decoder
    Multi-layer transformer decoder with vision-language communication
    """
    def __init__(self, 
                 d_model=256, 
                 nhead=8, 
                 num_decoder_layers=6, 
                 dim_feedforward=2048,
                 dropout=0.1):
        super(VimTSDecoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_decoder_layers
        
        # Multiple decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Learnable positional encoding for queries
        self.query_pos_embed = nn.Embedding(125, d_model)  # 125 = 100 det + 25 rec
        
    def forward(self, queries, visual_features, visual_pos=None):
        """
        Args:
            queries: [B, N, C] queries from Module 2
            visual_features: [B, H*W, C] flattened visual features from Module 1
            visual_pos: [B, H*W, C] positional encoding for visual features
            
        Returns:
            enhanced_queries: [B, N, C] refined queries
            all_attention_weights: list of attention maps from each layer
        """
        batch_size, num_queries, _ = queries.shape
        
        # Generate positional encoding for queries
        query_indices = torch.arange(num_queries, device=queries.device)
        query_pos = self.query_pos_embed(query_indices).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Process through decoder layers
        output = queries
        all_attention_weights = []
        
        for layer in self.layers:
            output, attn_weights = layer(output, visual_features, query_pos, visual_pos)
            all_attention_weights.append(attn_weights)
        
        # Final normalization
        output = self.norm(output)
        
        return output, all_attention_weights

class IntraGroupAttention(nn.Module):
    """
    Intra-group attention for detection and recognition queries
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super(IntraGroupAttention, self).__init__()
        
        self.detection_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.recognition_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm_det = nn.LayerNorm(d_model)
        self.norm_rec = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, num_detection_queries=100):
        """
        Args:
            queries: [B, N, C] combined detection and recognition queries
            num_detection_queries: number of detection queries
            
        Returns:
            enhanced_queries: [B, N, C] queries after intra-group attention
        """
        batch_size, total_queries, d_model = queries.shape
        
        # Split into detection and recognition queries
        det_queries = queries[:, :num_detection_queries, :]  # [B, 100, C]
        rec_queries = queries[:, num_detection_queries:, :]  # [B, 25, C]
        
        # Intra-group attention for detection queries
        det_enhanced, _ = self.detection_attn(det_queries, det_queries, det_queries)
        det_queries = det_queries + self.dropout(det_enhanced)
        det_queries = self.norm_det(det_queries)
        
        # Intra-group attention for recognition queries
        rec_enhanced, _ = self.recognition_attn(rec_queries, rec_queries, rec_queries)
        rec_queries = rec_queries + self.dropout(rec_enhanced)
        rec_queries = self.norm_rec(rec_queries)
        
        # Combine back
        enhanced_queries = torch.cat([det_queries, rec_queries], dim=1)
        
        return enhanced_queries

class InterGroupAttention(nn.Module):
    """
    Inter-group attention between detection and recognition queries
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super(InterGroupAttention, self).__init__()
        
        # Cross-attention between detection and recognition
        self.det_to_rec_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.rec_to_det_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm_det = nn.LayerNorm(d_model)
        self.norm_rec = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, num_detection_queries=100):
        """
        Args:
            queries: [B, N, C] combined detection and recognition queries
            num_detection_queries: number of detection queries
            
        Returns:
            enhanced_queries: [B, N, C] queries after inter-group attention
        """
        batch_size, total_queries, d_model = queries.shape
        
        # Split into detection and recognition queries
        det_queries = queries[:, :num_detection_queries, :]  # [B, 100, C]
        rec_queries = queries[:, num_detection_queries:, :]  # [B, 25, C]
        
        # Detection queries attend to recognition queries
        det_enhanced, _ = self.det_to_rec_attn(det_queries, rec_queries, rec_queries)
        det_queries = det_queries + self.dropout(det_enhanced)
        det_queries = self.norm_det(det_queries)
        
        # Recognition queries attend to detection queries  
        rec_enhanced, _ = self.rec_to_det_attn(rec_queries, det_queries, det_queries)
        rec_queries = rec_queries + self.dropout(rec_enhanced)
        rec_queries = self.norm_rec(rec_queries)
        
        # Combine back
        enhanced_queries = torch.cat([det_queries, rec_queries], dim=1)
        
        return enhanced_queries

# Complete Module 3 implementation
class CompleteVimTSDecoder(nn.Module):
    """
    Complete Module 3: Decoder with all components
    """
    def __init__(self, 
                 d_model=256,
                 nhead=8, 
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1):
        super(CompleteVimTSDecoder, self).__init__()
        
        # Main transformer decoder
        self.transformer_decoder = VimTSDecoder(
            d_model=d_model,
            nhead=nhead, 
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Intra and inter-group attention
        self.intra_group_attn = IntraGroupAttention(d_model, nhead, dropout)
        self.inter_group_attn = InterGroupAttention(d_model, nhead, dropout)
        
    def forward(self, queries, visual_features, visual_pos=None):
        """
        Args:
            queries: [B, N, C] queries from Module 2 (125 total)
            visual_features: [B, H*W, C] visual features from Module 1
            visual_pos: [B, H*W, C] positional encoding for features
            
        Returns:
            enhanced_queries: [B, N, C] final enhanced queries
            attention_weights: attention maps from transformer decoder
        """
        # Main transformer decoder processing
        decoder_output, attention_weights = self.transformer_decoder(
            queries, visual_features, visual_pos
        )
        
        # Intra-group attention (within detection/recognition groups)
        intra_enhanced = self.intra_group_attn(decoder_output)
        
        # Inter-group attention (between detection and recognition groups)  
        final_output = self.inter_group_attn(intra_enhanced)
        
        return final_output, attention_weights
