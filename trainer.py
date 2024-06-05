#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:23:30 2024

@author: mtross
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
# Fix for numpy boolean deprecation warning
np.bool = bool
import mxnet as mx
from mxnet import recordio
import json
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import time
from tqdm import tqdm
from collections import OrderedDict
from model import *
from utils import *


# Set manual seed for reproducibility
torch.manual_seed(10)


def train(**kwargs):
    """
    Trains the model using the provided hyperparameters.

    Parameters:
    **kwargs: Arbitrary keyword arguments containing training hyperparameters.
    """
    num_genes = kwargs['num_genes']
    num_val_genes = kwargs['num_val_genes']
    batch_size = kwargs['batch_size']
    patience = kwargs['patience']
    learning_rate = kwargs['learning_rate']
    data_dir = kwargs['data_dir']
    num_epochs = kwargs['max_epochs']
    max_seq_len = kwargs['seq_max_len']
    
    # Instantiate the model
    model = TransformerLayer().to(device)
    # model.load_state_dict(torch.load('/content/drive/MyDrive/maize_training_organs_famSplits/NEW_TISSUES/maize_03_03_2024__01_Epoch10.pth'))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer.load_state_dict(torch.load('/content/drive/MyDrive/maize_training_organs_famSplits/NEW_TISSUES/optimizer_03_01_2024__04.pth'))
    
    # File paths for training and validation data
    fname = f'{data_dir}/maize_recordio_train_ORGAN_RAW_pad15000_famsplits_NEWTiS_02_01_2024__15.rec'
    valid_fname = f'{data_dir}/maize_recordio_valid_ORGAN_RAW_pad15000_famsplits_NEWTiS_02_01_2024__15.rec'
    
    # Initialize variables for tracking training and validation loss
    training_loss = []
    validation_loss = []
    
    min_loss = 1000
    val_min_loss = 1000
    last_save = 0
    gene_loss = {}
    
    ts = datetime.now().strftime("%m_%d_%Y__%H")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        
        gene_models_seen = []
        num_gene_models_seen = 0
        total_loss = 0.0
        val_total_loss = 0.0
        record_file = recordio.MXRecordIO(fname, 'r')
        record_file_valid = recordio.MXRecordIO(valid_fname, 'r')
        
        for i in tqdm(range(num_genes), desc=f'Epoch {epoch}'):
            torch.cuda.empty_cache()
            record_file_read = record_file.read()
            gene_mod_data = json.loads(record_file_read.decode('utf-8'))
            gene = gene_mod_data['gene_mod']
            
            if len(gene_mod_data['seq']) > 90000:
                continue  # Skip due to memory issues
            if sum(gene_mod_data['expression']) == 0:
                continue  # Skip if there is no expression data
            if len(gene_mod_data['seq']) == 0:
                continue  # Skip if there is no sequence data
            
            gene_models_seen.append((gene, len(gene_mod_data['seq'])))
            gene_seq = torch.tensor(one_hot_encode(gene_mod_data['seq']), dtype=torch.float32)
            gene_expression = torch.tensor(max_normalize(gene_mod_data['expression']), dtype=torch.float32)
            
            num_gene_models_seen += 1
            my_input = gene_seq.to(device)
            labels = gene_expression.to(device)
            outputs = model(my_input).flatten()
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.cpu().item()
            gene_loss[gene] = loss.cpu().item()
            
            if i % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        avg_loss = total_loss / (num_gene_models_seen // batch_size)
        training_loss.append(avg_loss)
        print('Genes per epoch:', num_gene_models_seen)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}')
        
        record_file.close()
        
        ###################
        # Validation data #
        ###################
        
        model.eval()
        with torch.no_grad():
            for i2 in range(num_val_genes):
                record_file_valid_read = record_file_valid.read()
                if record_file_valid_read is None:
                    print('No more validation genes left on record.')
                    break
                gene_mod_data_val = json.loads(record_file_valid_read.decode('utf-8'))
                
                if sum(gene_mod_data_val['expression']) == 0:
                    continue
                if len(gene_mod_data_val['seq']) == 0:
                    continue  # Skip if there is no sequence data
                
                gene = gene_mod_data_val['gene_mod']
                gene_seq = torch.tensor(one_hot_encode(gene_mod_data_val['seq']), dtype=torch.float32)
                gene_expression = torch.tensor(max_normalize(gene_mod_data_val['expression']), dtype=torch.float32)
                
                if len(gene_mod_data_val['seq']) > 90000:
                    continue  # Skip due to memory issues
                
                my_input = gene_seq.to(device)
                labels = gene_expression.to(device)
                outputs = model(my_input).flatten()
                val_loss = criterion(outputs, labels)
                val_total_loss += val_loss
            
            val_avg_loss = val_total_loss / i2
            validation_loss.append(val_avg_loss)
            record_file_valid.close()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if val_avg_loss < val_min_loss:
            last_save = 0
            print('Validation genes per epoch:', i2)
            print(f'Min validation loss decreased from {val_min_loss} to {val_avg_loss}', 
                  f'Time elapsed: {round(elapsed_time, 2)} seconds')
            val_min_loss = val_avg_loss
            torch.save(model.state_dict(), f'{data_dir}/maize_{ts}_Epoch{epoch}_minValLoss.pth')
            torch.save(optimizer.state_dict(), f'{data_dir}/optimizer_{ts}.pth')
            continue
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'{data_dir}/maize_{ts}_Epoch{epoch}.pth')
        
        print('Validation loss:', val_avg_loss, f'Time elapsed: {round(elapsed_time, 2)} seconds')
        
        last_save += 1
        if last_save > patience:
            break
    
    pd.Series(training_loss).to_csv(f'{data_dir}/training_loss_{ts}_Epochs{epoch}.csv')
    pd.Series(validation_loss).to_csv(f'{data_dir}/validation_loss_{ts}_Epochs{epoch}.csv')
    
    torch.save(model.state_dict(), f'{data_dir}/maize_{ts}_Epoch{epoch}_minValLoss_Final.pth')
    torch.save(optimizer.state_dict(), f'{data_dir}/optimizer_{ts}.pth')

