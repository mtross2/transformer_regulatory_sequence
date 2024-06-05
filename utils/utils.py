#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:20:25 2024

@author: mtross
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot_encode(sequence):
    """
    One-hot encodes a DNA sequence.

    Parameters:
    sequence (str): The DNA sequence to encode.

    Returns:
    np.ndarray: One-hot encoded sequence.
    """
    nucleotides = ['A', 'T', 'C', 'G']
    encoding = {'A': [1, 0, 0, 0],
                'T': [0, 1, 0, 0],
                'C': [0, 0, 1, 0],
                'G': [0, 0, 0, 1],
                'N': [0, 0, 0, 0]}  # 'N' represents any nucleotide, encoded as all 0s

    encoded_sequence = []

    for nucleotide in sequence:
        if nucleotide in encoding:
            encoded_sequence.append(encoding[nucleotide])
        else:
            encoded_sequence.append(encoding['N'])

    return np.array(encoded_sequence)

class PoissonNegativeLogLikelihoodLoss(nn.Module):
    """
    Custom Poisson Negative Log-Likelihood Loss.
    """
    def __init__(self, reduction='mean'):
        """
        Initializes the loss function.

        Parameters:
        reduction (str): Specifies the reduction to apply to the output ('mean' or 'sum').
        """
        super(PoissonNegativeLogLikelihoodLoss, self).__init__()
        self.reduction = reduction

    def forward(self, predicted, target):
        """
        Computes the negative log-likelihood of the Poisson distribution.

        Parameters:
        predicted (torch.Tensor): Predicted values.
        target (torch.Tensor): Target values.

        Returns:
        torch.Tensor: Computed loss.
        """
        loss = torch.exp(-predicted) * target - predicted
        loss = loss + torch.log(torch.tensor(1e-10))  # Avoid numerical instability for log(0)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)

def min_max_normalize(data):
    """
    Applies min-max normalization to the data.

    Parameters:
    data (np.ndarray): Input data to normalize.

    Returns:
    np.ndarray: Normalized data.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val + 1e-10)
    return normalized_data

def max_normalize(data):
    """
    Applies max normalization to the data.

    Parameters:
    data (np.ndarray): Input data to normalize.

    Returns:
    np.ndarray: Normalized data.
    """
    max_val = np.max(data)
    normalized_data = data / (max_val + 1e-15)
    return normalized_data

def sigmoid(x):
    """
    Applies the sigmoid function.

    Parameters:
    x (np.ndarray): Input data.

    Returns:
    np.ndarray: Sigmoid of the input data.
    """
    return 1 / (1 + np.exp(-x))

def normalize_with_sigmoid(vector):
    """
    Normalizes the vector using the sigmoid function.

    Parameters:
    vector (np.ndarray): Input vector.

    Returns:
    np.ndarray: Sigmoid-normalized vector.
    """
    return sigmoid(vector)

