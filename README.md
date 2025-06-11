# TNMammo: A Multi-view Mammography Dataset for Breast Density Classification

[<p align="center"><img src="source/ex_mammo.png" width="500"></p>]()

## Description

This repository provides code for validating the quality and technical validity of the ThongNhat-Mammo dataset, a mammography image collection for breast density classification. The dataset can be downloaded from [our drive]().

## Features

- Multi-view mammogram analysis (CC and MLO views for both breasts)
- Breast density classification into 4 categories (A, B, C, D)
- Support for multiple datasets (DDSM, VinDr, ThongNhat)
- Cross-validation training support
- Comprehensive evaluation metrics
- Pre-processing pipeline including:
  - Breast region cropping
  - Image normalization
  - CLAHE enhancement
  - Resizing

## Installation

To install required packages:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized as follows:

```
datasets/
└── TNMammo/
    ├── images/
    │   └── ID/
    │       ├── left_cc.jpg
    │       ├── left_mlo.jpg
    │       ├── right_cc.jpg
    │       └── right_mlo.jpg
    └── TNMammo_labels.csv
```

## Usage
The dataset can be utilized for various machine learning and medical imaging applications, particularly in classification and diagnostic model development. Below are key considerations for usage:

- Preprocessing Recommendations: Images may require normalization and resizing to fit specific model input requirements.
- Potential Applications:
    + Supervised Learning: The dataset is well-suited for classification tasks where
      models learn to predict labels based on image features.
    + Feature Extraction: Researchers can extract and analyze specific features related to medical conditions.
    + Multi-View Analysis: Since each record contains four views, multi-view fusion techniques can be explored for improved accuracy.
- Considerations:
    + Ensure proper data splitting (e.g., train/validation/test sets) to avoid data leakage.
    + The dataset should be used responsibly in accordance with ethical and legal guidelines, especially for medical applications.
      
### Data Preprocessing

To preprocess the mammogram images:

1. Open preprocessing.ipynb in Jupyter Notebook
2. Follow the step-by-step instructions in the notebook to:
   - Load raw mammogram images
   - Apply breast region segmentation
   - Perform CLAHE enhancement
   - Normalize and resize images
   - Save processed images to the dataset structure

### Training

For single fold training:

```bash
python train.py --config_path ThongNhat_config.yml --fold 0
```

For training all folds:

```bash
python train_all_folds.py --config_path ThongNhat_config.yml --num_folds 5
```

### Evaluation

To evaluate the model:

```bash
python evaluate.py --config_path ThongNhat_config.yml
```

## Model Architecture

The model uses a ResNet50 backbone pretrained on ImageNet, with custom fully connected layers for breast density classification. The architecture includes:

- Multi-view feature extraction
- Feature fusion
- Dense classification layers

## Configuration

Model configuration can be modified through the `ThongNhat_config.yml` file, which includes settings for:

- Dataset parameters
- Model architecture
- Training hyperparameters
- Optimization settings
- Data augmentation

## License



## Citation

If you use this code or dataset in your research, please cite:

```bibtex
<!-- @article{thongnhat2024,
  title={ThongNhat-Mammo: A Multi-view Mammography Dataset for Breast Density Classification},
  author={[Author names]},
  journal={[Journal name]},
  year={2024}
} -->
```

## Acknowledgments

We would like to thank Thong Nhat Hospital for providing the mammography data used in this research. We would love to thank AISIA Research Lab, University of Science, Vietnam National University in Ho Chi Minh City, and Tan Tao University for supporting us in experiments and research algorithms for this project. 
