#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:55:17 2024

@author: mtross
"""
import argparse
import sys
import os
from predictor import *


# Parse command line arguments
parser = argparse.ArgumentParser(description='Predict gene expression and generate saliency map.')
parser.add_argument('--model', type=str, help='Path to saved model')
parser.add_argument('--sequence_file', help='Path to sequence text file')
parser.add_argument('--expression_file', help='Path to file with comma-separated expression values.\
                    This version of the software expects six comma-separated files representing the following tissues:\
                    \nLeaf, Embryo, Anther,Inflorescence,Endosperm, Root')

if __name__ == "__main__":
    # If the script is run directly (not imported as a module)
        
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Start prediction
    predict(model_path=args.model,
            sequence_file=args.sequence_file,
            expression_file=args.expression_file)

