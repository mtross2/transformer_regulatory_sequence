#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:23:30 2024

@author: mtross
"""
import argparse
import os
from trainer import train

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--data_dir', type=str, default=os.getcwd(), help='Path to the data directory')
    parser.add_argument('--max_epochs', type=int, default=2000, help='Maximum number of epochs for training')
    parser.add_argument('--seq_max_len', type=int, default=90000, help='Maximum length of sequences')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--learning_rate', type=float, default=0.000001, help='Learning rate for training the model')
    parser.add_argument('--patience', type=int, default=100, help='Number of epochs with no loss improvement before stopping training')
    parser.add_argument('--num_genes', type=int, default=28200, help='Number of genes for training')
    parser.add_argument('--num_val_genes', type=int, default=2000, help='Number of genes for validation')
    
    args = parser.parse_args()
    
    # Start training
    train(data_dir=args.data_dir, 
          max_epochs=args.max_epochs, 
          seq_max_len=args.seq_max_len, 
          batch_size=args.batch_size,
          num_gpus=args.num_gpus,
          learning_rate=args.learning_rate,
          patience=args.patience,
          num_genes=args.num_genes,
          num_val_genes=args.num_val_genes)
