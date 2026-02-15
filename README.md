# Casting Product Defect Detection System

A binary image classification system using deep learning to detect defects in casting products. The system uses transfer learning with a pretrained ResNet50 CNN model to classify products as either "OK" or "Defective".

## 🎯 Overview

This project implements an automated quality control system for industrial casting products. It uses computer vision and deep learning to identify defective products, helping to:
- Reduce manual inspection time
- Improve detection accuracy
- Maintain consistent quality standards
- Minimize human error

## 📊 Dataset

The system is designed for the [Real-Life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) from Kaggle.

**Dataset Structure:**
```
casting_data/
├── train/
│   ├── def_front/    # Defective products
│   └── ok_front/     # OK products
└── test/
    ├── def_front/    # Defective products
    └── ok_front/     # OK products
```

## 🏗️ Model Architecture

- **Base Model:** ResNet50 (pretrained on ImageNet)
- **Transfer Learning:** Feature extraction + fine-tuning
- **Custom Classification Head:**
  - Global Average Pooling
  - Dense layers (256, 128 neurons)
  - Batch Normalization
  - Dropout (0.5, 0.3)
  - Binary classification output (sigmoid)

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
cd casting_defect_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle:
   - Visit: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
   - Download and extract to `./casting_data/`

## 🚀 Usage

### Training the Model

Run the training script with default parameters:
```bash
python train_defect_classifier.py
```

**Training Process:**
1. Data preparation with augmentation
2. Initial training (transfer learning, frozen base)
3. Fine-tuning (unfrozen last 20 layers)
4. Evaluation on test set
5. Generation of visualizations

**Outputs:**
- `best_model.h5` - Best model during training
- `defect_classifier_model.h5` - Final trained model
- `training_history.png` - Training metrics over epochs
- `confusion_matrix.png` - Classification confusion matrix
- `roc_curve.png` - ROC curve with AUC score
- `predictions_sample.png` - Sample predictions visualization

### Making Predictions

#### Single Image Prediction
```bash
python predict_defects.py --model defect_classifier_model.h5 --image path/to/image.jpg --visualize
```

#### Batch Prediction
```bash
python predict_defects.py --model defect_classifier_model.h5 --image_dir path/to/images/ --visualize --output results.png
```

#### Command-line Arguments
- `--model`: Path to trained model file (default: `defect_classifier_model.h5`)
- `--image`: Path to single image for prediction
- `--image_dir`: Path to directory containing multiple images
- `--visualize`: Generate visualization of predictions
- `--output`: Path to save visualization output

## 📈 Performance Metrics

The model is evaluated using multiple metrics:
- **Accuracy:** Overall classification accuracy
- **Precision:** Proportion of correct positive predictions
- **Recall:** Proportion of actual positives correctly identified
- **AUC-ROC:** Area under the ROC curve
- **Confusion Matrix:** True/False positives and negatives

Expected performance (after training):
- Accuracy: ~98-99%
- Precision: ~98%
- Recall: ~98%
- AUC: ~0.99

## 🔬 Technical Details

### Data Augmentation
Training uses extensive augmentation to improve generalization:
- Random horizontal flipping
- Random rotation (±20°)
- Width/height shifting (±20%)
- Zoom (±20%)

### Training Strategy
1. **Phase 1 - Transfer Learning (30 epochs)**
   - Freeze ResNet50 base layers
   - Train only classification head
   - Learning rate: 0.0001
   
2. **Phase 2 - Fine-tuning (10 epochs)**
   - Unfreeze last 20 layers of ResNet50
   - Fine-tune with lower learning rate: 0.00001
   - Refine feature extraction for casting products

### Callbacks
- **Early Stopping:** Stops training if validation loss doesn't improve for 10 epochs
- **Learning Rate Reduction:** Reduces learning rate by 50% if validation loss plateaus
- **Model Checkpoint:** Saves best model based on validation accuracy

## 📁 Project Structure

```
casting_defect_detection/
│
├── train_defect_classifier.py    # Main training script
├── predict_defects.py             # Inference script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── casting_data/                  # Dataset directory (not included)
│   ├── train/
│   │   ├── def_front/
│   │   └── ok_front/
│   └── test/
│       ├── def_front/
│       └── ok_front/
│
└── outputs/                       # Generated during training
    ├── best_model.h5
    ├── defect_classifier_model.h5
    ├── training_history.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── predictions_sample.png
```

## 🎨 Customization

### Modify Image Size
```python
IMG_SIZE = (224, 224)  # Default ResNet50 input size
# or
IMG_SIZE = (256, 256)  # Higher resolution
```

### Adjust Batch Size
```python
BATCH_SIZE = 32  # Default
# Reduce if GPU memory is limited
BATCH_SIZE = 16
```

### Change Number of Epochs
```python
INITIAL_EPOCHS = 30  # Transfer learning phase
FINE_TUNE_EPOCHS = 10  # Fine-tuning phase
```

### Use Different Pretrained Models

Replace ResNet50 with other models:
```python
from tensorflow.keras.applications import VGG16, InceptionV3, EfficientNetB0

# VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))

# InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))

# EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
```

## 🔍 Understanding the Results

### Training History Plot
Shows how the model improves over epochs:
- **Accuracy/Loss:** Should improve/decrease steadily
- **Validation metrics:** Should track training metrics closely (no overfitting)

### Confusion Matrix
- **True Negatives (TN):** Correctly identified OK products
- **True Positives (TP):** Correctly identified defective products
- **False Negatives (FN):** Defective products missed (critical!)
- **False Positives (FP):** OK products incorrectly flagged as defective

### ROC Curve
- **AUC near 1.0:** Excellent discrimination between classes
- **AUC near 0.5:** Random classifier (poor performance)

## 🚨 Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` (try 16 or 8)
- Reduce `IMG_SIZE` to (128, 128)

### Low Accuracy
- Check dataset balance (equal OK/defective samples?)
- Increase number of training epochs
- Try different learning rates
- Add more data augmentation

### Model Overfitting
- Increase dropout rates
- Add more augmentation
- Reduce model complexity
- Use early stopping

### Training Too Slow
- Use GPU acceleration (CUDA-enabled GPU)
- Increase batch size (if memory allows)
- Reduce image size
- Use fewer training epochs

## 📚 References

- ResNet50 Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Transfer Learning: [A Comprehensive Hands-on Guide to Transfer Learning](https://arxiv.org/abs/1808.01974)
- Dataset: [Kaggle - Casting Product Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Suggestions and improvements are welcome! Consider:
- Testing with different pretrained models
- Implementing ensemble methods
- Adding explainability (Grad-CAM visualizations)
- Optimizing for edge deployment (TensorFlow Lite)

## 📧 Support

For issues or questions, please refer to the troubleshooting section or check the TensorFlow documentation.

---

**Happy Defect Detecting! 🔍✨**
