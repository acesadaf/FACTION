## Dependencies

### Primary Dependencies
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- fairlearn
- tqdm
- Python 3.6+


### Required Python Packages
```bash
pip install torch torchvision
pip install scikit-learn
pip install fairlearn
pip install tqdm
pip install numpy
pip install pandas
```


## Repository Components

### 1. Active Learning Implementation (active_learningFF.py)
- Handles dataset splitting and sampling
- Implements balanced sample acquisition
- Provides custom data samplers

### 2. Model Architecture (net folder)
- Provides implementations for the networks in the system.

### 3. Metrics and Evaluation (metrics folder)
- Uncertainty calculations
- Classification metrics
  - Demographic parity difference
  - Equal opportunity difference
  - Mutual information scoring

### 4. Utilities (utils folder)
- Gaussian Mixture Model implementation
- Embedding extraction
- Covariance computation
- default args
- train utilities (losses, training loops and other utils)

### 5. Training (train.py)
- The overall training code for the system

### 6. Datasets (datasets.py)
- Provides Dataset classes for each of the datasets used in FACTION.

## Running the Project

### Command Line Arguments


#### Required Arguments
- `--seed`: Random seed 

#### Optional Arguments
- `--MU`: Fairness regularization parameter (default: 0.7)
- `--slack`: Slack parameter (default: 0.1)
- `--LAM`: Fair selection parameter (default: 1.0)
- `--model`: Model architecture (default: "resnet18")
- `--epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 0.1)
- `--train_batch_size`: Training batch size (default: 64)
- `--test_batch_size`: Testing batch size (default: 512)
- `--scoring_batch_size`: Batch size for scoring (default: 128)
- `--num_initial_samples`: Initial training set size (default: 100)
- `--max_training_samples`: Maximum training samples (default: 200)
- `--acquisition_batch_size`: Samples to acquire per step (default: 50)
- `--sn`: Enable spectral normalization
- `--dataset`: Dataset to use (default: "CelebA", choices: "RotatedColoredMNIST", "FairFace", "CelebA", "FFHQ", "NYSF")

### Data

Please ensure you download the required datasets. Certain datasets like FairFace are substantially large and will require several gigabytes of disk space. Once the data is downloaded, please update the paths in the respective dataset classes in datasets.py (e.g. self.dir = os.path.join("/path/to/data")) so that the code can read it from the correct location. 

### Running the Script

To run the training script:

```bash
python train.py --seed <value> [other arguments]
```

### Example Usage

```bash
python train.py --seed 42 --MU 0.1 --slack 0.05 --LAM 1.0 --epochs 100 --lr 0.001
```
