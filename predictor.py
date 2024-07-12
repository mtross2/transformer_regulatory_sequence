#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:10:31 2024

@author: mtross
"""
import sys
try:
    import numpy as np
except:
    np.bool = bool
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
np.bool = bool
import mxnet as mx
from mxnet import recordio
import json
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats
import time
import pickle
from tqdm import tqdm
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr
from model import *
from utils import *

ts = datetime.now().strftime("%m_%d_%Y__%H")
torch.manual_seed(10)


def predict(**kwargs):
    """
    Predict gene expression and generate saliency map.

    Parameters:
        kwargs (dict): Dictionary containing the following keys:
            - model_path (str): Path to the pre-trained model.
            - sequence_file (str): Path to the file containing gene sequence.
            - expression_file (str): Path to the file containing gene expression data.
    
    Returns:
        None
    """
    # Function to predict gene expression and saliency based on provided inputs
    
    # Unpack input arguments
    model_path = kwargs['model_path']
    sequence_path = kwargs['sequence_file']
    expression_path = kwargs['expression_file']
    
    # Prompt user for sequence orientation
    print("Ensure that gene is padded with 15,000 bp upstream of the transcription start site \
          and 15,000 bp downstream of the transcription end site")
    forward = input('Is sequence on the forward strand (y/n):')
    
    # Read sequence from file
    with open(sequence_path, 'r') as f:
        seq = f.read()
        
    print(f'Padded sequence length = {len(seq)}')
    
    # Ensure user provides valid input for sequence orientation
    assert forward in ['y', 'n'],'Indicate yes with "y" and no with "n"'
    
    # Read expression data from file
    with open(expression_path, 'r') as f2:
        expression_file = f2.read()
        
    # Convert expression data to list of integers
    expression = [int(x) for x in expression_file.split(",")]
    
    # Ensure there are six expression values provided
    assert len(expression) == 6, 'This implementation of the software expects six expression values in the order of the following tissues types:\
                                 \nLeaf, Embryo, Anther,Inflorescence,Endosperm, Root'
                                
    print(f'Expression values: {expression}')
    
    # Normalize expression values to relative expression
    expression = max_normalize(expression)
   
    # Reverse sequence if it's on the reverse strand
    if forward == 'n':
        seq = seq[::-1]
    
    # Define loss function
    criterion = torch.nn.L1Loss()
    
    # Load pre-trained model
    model = customTransformer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.position_encoding.requires_grad = False
    model.eval()
    
    # Encode gene sequence into one-hot format
    gene_seq = torch.tensor(one_hot_encode(seq), dtype=torch.float32)
    
    # Normalize expression
    gene_expression = torch.tensor(max_normalize(expression))
    
    # Move inputs to GPU if available
    my_input = gene_seq.to(device)
    my_input.requires_grad = True
    
    # Forward pass to get model predictions
    outputs = model(my_input).flatten()
        
    # Zero gradients
    model.zero_grad()
    
    # Convert expression data to tensor
    labels = torch.Tensor(expression)
    
    # Compute loss
    loss = criterion(outputs, labels.unsqueeze(0))
    
    # Backpropagate the loss
    loss.backward()
          
    # Calculate saliency
    saliency = my_input.grad.abs() # Do the absolute values?
    saliency_avg = torch.mean(saliency, dim=1).cpu().numpy()
    
    # Save saliency scores to a CSV file
    pd.Series(saliency_avg).to_csv(f'saliency_raw__{ts}.csv', index=False)
    
    # Plot saliency scores
    fig = plt.figure(figsize= (10,5))
    ax = fig.add_subplot(111)
    max_normalized_saliency = list(saliency_avg)/max(list(saliency_avg))
    sns.scatterplot(max_normalized_saliency, ax = ax)
    ax.vlines(x= 15000, ymin = 0, ymax = 1, colors = 'red', linestyles = '--')
    ax.vlines(x= len(saliency_avg)-15000, ymin = 0, ymax = 1, colors = 'red', linestyles = '--')
    ax.set_xlabel('Nucleotide position',labelpad = 10)
    ax.set_ylabel('Max normalized \n saliency score', labelpad = 10)
    ax.set_ylim(0,1.1)
    ax.set_xlim(0, len(seq))
    fig.tight_layout()
    
    # Save plot as image
    plt.savefig(f'saliency__{ts}.png', dpi=300)
    plt.show()

