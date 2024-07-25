Custom transformer model

Overview

* This project includes a pytorch implementation of a customer transformer model that takes as input contextual sequence for genes of interest, predict relative gene expression and produce saliency for maps for the input sequence.
* Approach implemented in "Tross, M. C, Grzybowski, M. W, Jubery, T. Z, Grove, R. J, Nishimwe, A. V, Torres-Rodriguez, J. V., Sun, G., Ganapathysubramanian, B., Ge, Y., & Schnable, J. C. (2024). Data driven discovery and quantification of hyperspectral leaf reflectance phenotypes across a maize diversity panel. The Plant Phenome Journal, 7, e20106. https://doi.org/10.1002/ppj2.20106"


Getting Started
Prerequisites

* Python 3.x
* Pip package manager

Installation

1. Clone the Repository
    * Run the following commands:
	```{bash}
    	git clone https://github.com/mtross2/transformer_regulatory_sequence.git
	```
	```{bash}
    	cd transformer_regulatory_sequence
	```
2. Set Up a Virtual Environment (Optional but recommended)

    * For Windows:
	```{bash}
        python -m venv venv
	```
	```{bash}
        .\venv\Scripts\activate
	```
    * For Unix or MacOS:
    	```{bash}
        python3 -m venv venv
	```
	```{bash}
        source venv/bin/activate
	```
3. Install Required Packages

    * Execute the command:
	```{bash}
        pip install -r requirements.txt
	```
4. Install Your Package (Optional if you want to use it as a package)

    * Use this command:
	```{bash}
        python setup.py install
	```
# Training Gene Expression Prediction Model

This script (`train.py`) is used to train a deep learning model for predicting gene expression levels based on gene sequences.

## Usage

To train the model, simply run the script with the necessary arguments:

```{bash}
python train.py --data_dir /path/to/data --max_epochs 2000 --seq_max_len 90000 --batch_size 1 --num_gpus 1 --learning_rate 0.000001 --patience 100 --num_genes 28200 --num_val_genes 2000
```

## Replace /path/to/data with the directory containing your training data.
Arguments

    data_dir: Path to the data directory containing training data.
    max_epochs: Maximum number of epochs for training.
    seq_max_len: Maximum length of gene sequences.
    batch_size: Batch size for training.
    num_gpus: Number of GPUs to use.
    learning_rate: Learning rate for training the model.
    patience: Number of epochs with no loss improvement before stopping training.
    num_genes: Number of genes for training.
    num_val_genes: Number of genes for validation.

# Gene Saliency Prediction

This script (`predict.py`) is used to predict gene expression levels and generate saliency maps based on provided gene sequences.

# Usage

# To predict gene expression, run the script with the following arguments:

```{bash}
python predict.py --model /path/to/model.pth --sequence_file /path/to/sequence.txt --expression_file /path/to/expression.txt
```

# Replace /path/to/model.pth, /path/to/sequence.txt, and /path/to/expression.txt with the paths to your trained model, gene sequence file, and expression file, respectively.
# Arguments

    model: Path to the saved model file.
    sequence_file: Path to the file containing gene sequence.
    expression_file: Path to the file containing gene expression data.

# License

This project is licensed under the CC-BY-NC  License. See the LICENSE file for details.
