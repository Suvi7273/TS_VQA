# VimTS Module 4: PQGM (Prompt Query Generation Module)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class PromptEmbedding(nn.Module):
    """
    Learnable prompt embeddings for different granularities
    """
    def __init__(self, d_model=256, max_prompt_len=50):
        super(PromptEmbedding, self).__init__()
        
        self.d_model = d_model
        self.max_prompt_len = max_prompt_len
        
        # Different granularity prompts
        self.character_prompts = nn.Parameter(torch.randn(max_prompt_len, d_model) * 0.02)
        self.word_prompts = nn.Parameter(torch.randn(max_prompt_len, d_model) * 0.02)  
        self.line_prompts = nn.Parameter(torch.randn(max_prompt_len, d_model) * 0.02)
        
        # Prompt selection network
        self.prompt_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),  # 3 granularities
            nn.Softmax(dim=-1)
        )
        
        # Prompt fusion network
        self.prompt_fusion = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, granularity_hints=None):
        """
        Args:
            queries: [B, N, C] detection/recognition queries
            granularity_hints: [B, N, 3] optional granularity preferences
            
        Returns:
            prompt_enhanced_queries: [B, N, C] enhanced queries with prompts
            prompt_weights: [B, N, 3] granularity selection weights
        """
        batch_size, num_queries, d_model = queries.shape
        
        # Get prompt selection weights
        if granularity_hints is not None:
            prompt_weights = granularity_hints
        else:
            prompt_weights = self.prompt_selector(queries)  # [B, N, 3]
        
        # Prepare prompts for each batch and query
        enhanced_queries = []
        
        for b in range(batch_size):
            batch_enhanced = []
            
            for q in range(num_queries):
                query = queries[b, q].unsqueeze(0)  # [1, C]
                weights = prompt_weights[b, q]  # [3]
                
                # Weighted combination of prompts
                char_weight, word_weight, line_weight = weights[0], weights[1], weights[2]
                
                combined_prompts = (char_weight * self.character_prompts + 
                                  word_weight * self.word_prompts + 
                                  line_weight * self.line_prompts)  # [max_prompt_len, C]
                
                # Select relevant prompts (top-k)
                prompt_len = min(10, self.max_prompt_len)
                selected_prompts = combined_prompts[:prompt_len].unsqueeze(0)  # [1, prompt_len, C]
                
                # Fuse query with prompts using cross-attention
                enhanced_query, _ = self.prompt_fusion(
                    query.unsqueeze(1),  # [1, 1, C] as query
                    selected_prompts,    # [1, prompt_len, C] as key
                    selected_prompts     # [1, prompt_len, C] as value
                )
                
                enhanced_query = enhanced_query.squeeze(1)  # [1, C]
                
                # Residual connection and normalization
                final_query = self.norm(query.squeeze(0) + enhanced_query.squeeze(0))
                
                batch_enhanced.append(final_query)
            
            enhanced_queries.append(torch.stack(batch_enhanced))
        
        prompt_enhanced_queries = torch.stack(enhanced_queries)  # [B, N, C]
        
        return prompt_enhanced_queries, prompt_weights

class MultiGranularityAttention(nn.Module):
    """
    Multi-granularity attention for different text levels
    """
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        super(MultiGranularityAttention, self).__init__()
        
        # Separate attention heads for different granularities
        self.char_attention = MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.word_attention = MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.line_attention = MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Granularity fusion
        self.granularity_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, visual_features, granularity_weights):
        """
        Args:
            queries: [B, N, C] input queries
            visual_features: [B, H*W, C] visual features
            granularity_weights: [B, N, 3] granularity preferences
            
        Returns:
            multi_granular_queries: [B, N, C] enhanced queries
        """
        batch_size, num_queries, d_model = queries.shape
        
        # Apply different attention mechanisms
        char_attended, _ = self.char_attention(queries, visual_features, visual_features)
        word_attended, _ = self.word_attention(queries, visual_features, visual_features)  
        line_attended, _ = self.line_attention(queries, visual_features, visual_features)
        
        # Weight by granularity preferences
        char_weights = granularity_weights[:, :, 0:1]  # [B, N, 1]
        word_weights = granularity_weights[:, :, 1:2]  # [B, N, 1]
        line_weights = granularity_weights[:, :, 2:3]  # [B, N, 1]
        
        weighted_char = char_attended * char_weights
        weighted_word = word_attended * word_weights
        weighted_line = line_attended * line_weights
        
        # Concatenate and fuse
        multi_granular = torch.cat([weighted_char, weighted_word, weighted_line], dim=-1)
        fused_queries = self.granularity_fusion(multi_granular)
        
        # Residual connection and normalization
        output = self.norm(queries + self.dropout(fused_queries))
        
        return output

class CrossDomainAdapter(nn.Module):
    """
    Cross-domain adaptation component
    """
    def __init__(self, d_model=256, num_domains=5, dropout=0.1):
        super(CrossDomainAdapter, self).__init__()
        
        self.d_model = d_model
        self.num_domains = num_domains
        
        # Domain-specific adapters
        self.domain_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model)
            ) for _ in range(num_domains)
        ])
        
        # Domain classifier (for training)
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_domains)
        )
        
        # Adaptive weighting
        self.domain_weighting = nn.Sequential(
            nn.Linear(d_model, num_domains),
            nn.Softmax(dim=-1)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, domain_id=None, training=True):
        """
        Args:
            queries: [B, N, C] input queries
            domain_id: int, specific domain ID (if known)
            training: bool, whether in training mode
            
        Returns:
            adapted_queries: [B, N, C] domain-adapted queries
            domain_logits: [B, N, num_domains] domain classification logits
        """
        batch_size, num_queries, d_model = queries.shape
        
        # Get domain weights
        domain_weights = self.domain_weighting(queries)  # [B, N, num_domains]
        
        # Apply domain-specific adapters
        adapted_outputs = []
        for domain_idx in range(self.num_domains):
            adapted = self.domain_adapters[domain_idx](queries)  # [B, N, C]
            adapted_outputs.append(adapted)
        
        adapted_stack = torch.stack(adapted_outputs, dim=-1)  # [B, N, C, num_domains]
        
        # Weighted combination of domain adaptations
        domain_weights_expanded = domain_weights.unsqueeze(-2)  # [B, N, 1, num_domains]
        adapted_queries = torch.sum(adapted_stack * domain_weights_expanded, dim=-1)  # [B, N, C]
        
        # Residual connection
        adapted_queries = self.norm(queries + adapted_queries)
        
        # Domain classification for training
        domain_logits = None
        if training:
            domain_logits = self.domain_classifier(adapted_queries)
        
        return adapted_queries, domain_logits

class PQGM(nn.Module):
    """
    Complete Prompt Query Generation Module (Module 4)
    """
    def __init__(self, 
                 d_model=256,
                 num_heads=8,
                 num_domains=5,
                 max_prompt_len=50,
                 dropout=0.1):
        super(PQGM, self).__init__()
        
        # Core components
        self.prompt_embedding = PromptEmbedding(d_model, max_prompt_len)
        self.multi_granularity_attention = MultiGranularityAttention(d_model, num_heads, dropout)
        self.cross_domain_adapter = CrossDomainAdapter(d_model, num_domains, dropout)
        
        # Final enhancement layer
        self.final_enhancement = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, visual_features, domain_id=None, granularity_hints=None, training=True):
        """
        Args:
            queries: [B, N, C] input queries from Module 3
            visual_features: [B, H*W, C] visual features from Module 1
            domain_id: int, domain identifier (optional)
            granularity_hints: [B, N, 3] granularity preferences (optional)
            training: bool, training mode flag
            
        Returns:
            enhanced_queries: [B, N, C] prompt-enhanced queries
            pqgm_outputs: dict with additional outputs for analysis/loss
        """
        # Step 1: Prompt embedding and enhancement
        prompt_queries, prompt_weights = self.prompt_embedding(queries, granularity_hints)
        
        # Step 2: Multi-granularity attention
        multi_granular_queries = self.multi_granularity_attention(
            prompt_queries, visual_features, prompt_weights
        )
        
        # Step 3: Cross-domain adaptation
        adapted_queries, domain_logits = self.cross_domain_adapter(
            multi_granular_queries, domain_id, training
        )
        
        # Step 4: Final enhancement
        enhanced = self.final_enhancement(adapted_queries)
        enhanced_queries = self.norm(adapted_queries + enhanced)
        
        # Prepare additional outputs
        pqgm_outputs = {
            'prompt_weights': prompt_weights,
            'domain_logits': domain_logits,
            'granularity_distribution': prompt_weights.mean(dim=1)  # [B, 3]
        }
        
        return enhanced_queries, pqgm_outputs
