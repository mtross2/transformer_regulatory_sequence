#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 9 14:17:46 2024

@author: mtross
"""

import torch
import torch.nn as nn
import math

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Positional Encoding module for adding positional information to the input
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.01, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:, :x.size(0)]
        return self.dropout(x)

# Feed Forward module
class FeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super(FeedForward, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, x):
        # Forward pass through the feed forward network
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Transformer Layer module
# Can replace attenion modules with tranformer layer module
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(TransformerLayer, self).__init__()

        # Multi-head self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Feed Forward network
        self.feed_forward = FeedForward(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Self-attention mechanism
        attended, _ = self.attention(x, x, x)
        x = x + attended
        x = self.norm1(x)

        # Feed Forward network
        fed_forward = self.feed_forward(x.transpose(0, 1))
        x = x + fed_forward
        x = self.norm2(x)
        return x

# Transformer module
class customTransformer(nn.Module):
    def __init__(self):
        super(customTransformer, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(4, 1000, kernel_size=15, padding='valid')
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv1d(1000, 500, kernel_size=15, padding='same')
        self.gelu2 = nn.GELU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(500, 250, kernel_size=15, padding='same')
        self.gelu3 = nn.GELU()
        self.conv4 = nn.Conv1d(250, 500, kernel_size=15, padding='same')
        self.gelu4 = nn.GELU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv1d(500, 500, kernel_size=15, padding='same')
        self.gelu5 = nn.GELU()
        self.conv6 = nn.Conv1d(500, 1000, kernel_size=15, padding='same')
        self.gelu6 = nn.GELU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)

        # Layer normalization and positional encoding
        self.norm = nn.LayerNorm(1000)
        self.position_encoding = PositionalEncoding(d_model=1000, max_len=30000)

        # Multi-head self-attention mechanism
        self.multihead1 = nn.MultiheadAttention(embed_dim=1000, num_heads=10)
        self.multihead2 = nn.MultiheadAttention(embed_dim=1000, num_heads=10)
        self.multihead3 = nn.MultiheadAttention(embed_dim=1000, num_heads=10)
        self.multihead4 = nn.MultiheadAttention(embed_dim=1000, num_heads=10)
        self.multihead5 = nn.MultiheadAttention(embed_dim=1000, num_heads=10)

        # Fully connected layers
        self.fc1 = nn.Linear(1000, 1000 * 4)
        self.gelu7 = nn.GELU()
        self.dropout4 = nn.Dropout(0.25)
        self.dropout5 = nn.Dropout(0.25)
        self.dropout6 = nn.Dropout(0.25)
        self.dropout7 = nn.Dropout(0.25)
        self.dropout8 = nn.Dropout(0.25)
        self.dropout9 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(1000 * 4, 1000)
        self.gelu8 = nn.GELU()
        self.fc3 = nn.Linear(1000, 6)
        self.gelu9 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Convolutional layers
        x = x.transpose(0, 1)
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.gelu2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.gelu3(x)
        x = self.conv4(x)
        x = self.gelu4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.gelu5(x)
        x = self.conv6(x)
        x = self.gelu6(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        x = x.transpose(0, 1)

        # Positional Encoding
        x1 = self.position_encoding(x)

        # Multi-head self-attention mechanisms
        x, _ = self.multihead1(x1, x1, x1)
        x = self.norm(x + x1)
        x = self.dropout5(x)
        
        x, _ = self.multihead2(x, x, x)
        x = self.norm(x + x1)
        x = self.dropout6(x)
        
        x, _ = self.multihead3(x, x, x)
        x = self.norm(x + x1)
        x = self.dropout7(x)
        
        x, _ = self.multihead4(x, x, x)
        x = self.norm(x + x1)
        x = self.dropout8(x)
        
        x, _ = self.multihead5(x, x, x)
        x = self.norm(x + x1)
        x = self.dropout9(x)
        
        # Global average pooling
        x = x.squeeze(0)
        x = x.transpose(0, 1)
        x = self.pool(x)
        x = x.transpose(0, 1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.gelu7(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.gelu8(x)
        x = self.fc3(x)
        
        return x
