"""
Transformer-based policy networks for the temporal GFN.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalTransformerModel(nn.Module):
    """
    Transformer-based model for temporal GFN policies.
    
    This model can be used for both forward and backward policies. The forward policy
    predicts the next quantized value given the context and partial forecast. The
    backward policy predicts which position to modify given the full forecast.
    
    Attributes:
        d_model (int): Dimension of the model
        nhead (int): Number of attention heads
        d_hid (int): Dimension of the feedforward network
        nlayers (int): Number of transformer layers
        dropout (float): Dropout probability
        k (int): Number of quantization bins (action space size)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        d_hid: int = 512,
        nlayers: int = 4,
        dropout: float = 0.1,
        k: int = 100,
        context_length: int = 96,
        prediction_horizon: int = 24,
        is_forward: bool = True,
        uniform_init: bool = False,
    ):
        """
        Initialize the transformer model.
        
        Args:
            d_model (int): Dimension of the model
            nhead (int): Number of attention heads
            d_hid (int): Dimension of the feedforward network
            nlayers (int): Number of transformer layers
            dropout (float): Dropout probability
            k (int): Number of quantization bins (action space size)
            context_length (int): Length of the context window
            prediction_horizon (int): Number of future steps to predict
            is_forward (bool): Whether this is a forward or backward policy
            uniform_init (bool): Whether to use uniform initialization for output layers
        """
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.k = k
        self.is_forward = is_forward
        
        # Input embedding layers
        self.value_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Time step embedding (position in the sequence)
        self.time_embedding = nn.Embedding(context_length + prediction_horizon, d_model)
        
        # Mask embedding (for distinguishing real vs. predicted values)
        self.mask_embedding = nn.Embedding(2, d_model)
        
        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        # Output layers
        if is_forward:
            # Forward policy outputs logits for each possible quantized value
            self.output_layer = nn.Linear(d_model, k)
        else:
            # Backward policy outputs logits for positions to modify
            self.output_layer = nn.Linear(d_model, prediction_horizon)
        
        # Initialize with uniform weights for the output layer if specified
        if uniform_init:
            if is_forward:
                # Initialize to predict uniform distribution over action space
                self.output_layer.weight.data.fill_(0.0)
                self.output_layer.bias.data.fill_(0.0)
            else:
                # Initialize to predict uniform distribution over positions
                self.output_layer.weight.data.fill_(0.0)
                self.output_layer.bias.data.fill_(0.0)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize non-output layer weights."""
        initrange = 0.1
        self.value_embedding.weight.data.uniform_(-initrange, initrange)
        
    def update_action_space(self, new_k: int):
        """
        Update the action space size (number of quantization bins).
        This is used for adaptive quantization.
        
        Args:
            new_k (int): New number of quantization bins
        """
        if not self.is_forward or new_k == self.k:
            return  # Only update if this is a forward policy and K has changed
        
        old_k = self.k
        self.k = new_k
        
        # Create a new output layer with the updated size
        old_layer = self.output_layer
        new_layer = nn.Linear(self.d_model, new_k)
        
        # If the new action space is larger, copy existing weights and initialize new ones
        if new_k > old_k:
            # Copy existing weights
            new_layer.weight.data[:old_k, :] = old_layer.weight.data
            new_layer.bias.data[:old_k] = old_layer.bias.data
            
            # Initialize new weights around the same range as existing ones
            weight_std = old_layer.weight.data.std().item()
            bias_mean = old_layer.bias.data.mean().item()
            
            new_layer.weight.data[old_k:, :].normal_(0.0, weight_std)
            new_layer.bias.data[old_k:].fill_(bias_mean)
        else:
            # If smaller, just copy the subset of weights we need
            new_layer.weight.data = old_layer.weight.data[:new_k, :]
            new_layer.bias.data = old_layer.bias.data[:new_k]
        
        # Replace the output layer
        self.output_layer = new_layer
    
    def forward(
        self,
        context: torch.Tensor,
        forecast: torch.Tensor,
        forecast_mask: torch.Tensor,
        step: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            context: Tensor of shape [batch_size, context_length] containing context window
            forecast: Tensor of shape [batch_size, prediction_horizon] containing partial forecast
            forecast_mask: Boolean tensor of shape [batch_size, prediction_horizon] indicating 
                          which forecast positions have been filled
            step: Current step in the prediction sequence (for forward policy)
            
        Returns:
            logits: 
                - For forward policy: action logits of shape [batch_size, k]
                - For backward policy: position logits of shape [batch_size, prediction_horizon]
        """
        batch_size = context.shape[0]
        device = context.device
        
        # Create the combined sequence: context + (partial) forecast
        # Replace NaN values with 0 to avoid computation issues
        forecast_cleaned = torch.where(
            torch.isnan(forecast),
            torch.zeros_like(forecast),
            forecast
        )
        
        combined = torch.cat([
            context.unsqueeze(-1),  # [batch_size, context_length, 1]
            forecast_cleaned.unsqueeze(-1)  # [batch_size, prediction_horizon, 1]
        ], dim=1)  # [batch_size, context_length + prediction_horizon, 1]
        
        # Create the mask tensor (0 for context, 1 for forecast)
        mask_indices = torch.cat([
            torch.zeros(batch_size, self.context_length, device=device, dtype=torch.long),
            torch.ones(batch_size, self.prediction_horizon, device=device, dtype=torch.long)
        ], dim=1)  # [batch_size, context_length + prediction_horizon]
        
        # Create time step indices
        time_indices = torch.arange(
            self.context_length + self.prediction_horizon, 
            device=device
        ).expand(batch_size, -1)  # [batch_size, context_length + prediction_horizon]
        
        # Embed the values
        value_embeddings = self.value_embedding(combined)
        
        # Add time embeddings
        time_embeddings = self.time_embedding(time_indices)  # [batch_size, seq_len, d_model]
        embeddings = value_embeddings + time_embeddings
        
        # Add mask embeddings to differentiate context from forecast
        mask_embeddings = self.mask_embedding(mask_indices)  # [batch_size, seq_len, d_model]
        embeddings = embeddings + mask_embeddings
        
        # Create attention mask to prevent attending to future forecast values
        # For the forward policy, we should only allow attending to context and past forecast values
        if self.is_forward and step is not None:
            attn_mask = torch.ones(
                self.context_length + self.prediction_horizon,
                self.context_length + self.prediction_horizon,
                device=device
            )
            # Allow attending to context and past forecast values
            forecast_start_idx = self.context_length
            # Can attend to all context
            attn_mask[:, :forecast_start_idx] = 0
            # Can attend to past forecast values
            past_forecast_end = forecast_start_idx + step
            attn_mask[:, forecast_start_idx:past_forecast_end] = 0
            # Future forecast values are masked (kept as 1)
            attn_mask = attn_mask.bool()
        else:
            attn_mask = None
        
        # Apply transformer
        transformer_output = self.transformer_encoder(
            embeddings, mask=attn_mask
        )  # [batch_size, seq_len, d_model]
        
        if self.is_forward:
            # For forward policy, we're predicting the next value
            if step is None:
                # If step is not specified, use the last unfilled position
                # based on the forecast mask
                unfilled_pos = (~forecast_mask).long().argmax(dim=1)
                idx = self.context_length + unfilled_pos
            else:
                # Otherwise use the specified step
                idx = self.context_length + step
            
            # Get the representation at the current forecast position
            batch_indices = torch.arange(batch_size, device=device)
            relevant_output = transformer_output[batch_indices, idx]  # [batch_size, d_model]
            
            # Compute logits for each possible action (quantized value)
            logits = self.output_layer(relevant_output)  # [batch_size, k]
        else:
            # For backward policy, we're predicting which position to modify
            # Only use the forecast part of the sequence
            forecast_output = transformer_output[:, self.context_length:]  # [batch_size, pred_horizon, d_model]
            
            # Use the representation of each position to predict modification probabilities
            # First, get a summary of the entire forecast by mean pooling
            forecast_summary = torch.mean(forecast_output, dim=1)  # [batch_size, d_model]
            
            # Compute logits for each position
            logits = self.output_layer(forecast_summary)  # [batch_size, prediction_horizon]
            
            # Mask out positions that are not filled (can't modify what isn't there)
            if forecast_mask is not None:
                mask = ~forecast_mask  # Positions that are NOT filled
                logits = logits.masked_fill(mask, float('-inf'))
        
        return logits 