# Vegetable Classifier

A deep learning image classification system for identifying 15 different vegetables using PyTorch and a fine-tuned ResNet50 model.

## Project Overview

This project implements a vegetable classification system that can identify the following 15 vegetable categories:
- Bean
- Bitter Gourd
- Bottle Gourd
- Brinjal (Eggplant)
- Broccoli
- Cabbage
- Capsicum (Bell Pepper)
- Carrot
- Cauliflower
- Cucumber
- Papaya
- Potato
- Pumpkin
- Radish
- Tomato

The model achieves over 97% accuracy on the test dataset using transfer learning with a pre-trained ResNet50 architecture.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- Pillow (PIL)
- matplotlib
- numpy
- scikit-learn
- tqdm

You can install the required packages using:

```bash
pip install torch torchvision pillow matplotlib numpy scikit-learn tqdm
```

## Dataset Structure

The dataset should be organized in the following directory structure:

```
dataset/
├── train/
│   ├── bean/
│   ├── bitter_gourd/
│   └── ...
├── validation/
│   ├── bean/
│   ├── bitter_gourd/
│   └── ...
└── test/
    ├── bean/
    ├── bitter_gourd/
    └── ...
```

Each subfolder should contain the respective vegetable images in JPG, JPEG, or PNG format.

## Usage

### Training the Model

```python
from vegetable_classifier import run_vegetable_classifier

# Train the model with default parameters
classifier = run_vegetable_classifier(
    data_dir='dataset',  # Path to your dataset
    num_epochs=5,
    batch_size=16
)
```

### Using a Pre-trained Model

```python
from vegetable_classifier import VegetableClassifier

# Initialize the classifier
classifier = VegetableClassifier(num_classes=15)

# Load a pre-trained model
classifier.load_model('best_vegetable_model.pth')

# Predict a single image
class_name, confidence, top3_predictions = classifier.predict('path/to/image.jpg')
print(f"Predicted: {class_name} with {confidence:.2%} confidence")
print("Top 3 predictions:")
for veg, conf in top3_predictions:
    print(f"- {veg}: {conf:.2%}")
```

### Visualizing Predictions

```python
from vegetable_classifier import visualize_prediction

# Visualize the prediction for a specific image
visualize_prediction('path/to/image.jpg', classifier)
```

## Model Architecture

This project uses a ResNet50 model pre-trained on ImageNet, with the final fully connected layer modified to output 15 classes (one for each vegetable type). The architecture employs:

- Transfer learning from a pre-trained ResNet50 model
- Adam optimizer with a learning rate of 0.001
- Cross-entropy loss function
- Data augmentation for training (random crops, horizontal flips)

## Performance

The model achieves the following performance on the test dataset:

- Test Accuracy: 97.60%
- A confusion matrix visualization is generated to show the performance across different vegetable classes

## Project Files

- `vegetable_classifier.py`: Contains the main `VegetableClassifier` class and utility functions
- `best_vegetable_model.pth`: The saved model weights after training
- `confusion_matrix.png`: Visualization of model performance on the test set

## Data Preprocessing

Images are preprocessed using the following transforms:

### Training
- Random resized crop to 224x224 pixels
- Random horizontal flip
- Normalization with ImageNet mean and standard deviation

### Validation/Testing
- Resize to 256 pixels
- Center crop to 224x224 pixels
- Normalization with ImageNet mean and standard deviation

## Model Training

The model is trained with the following default parameters:
- Batch size: 16
- Number of epochs: 5
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Cross-Entropy Loss

Training progress is monitored using validation accuracy, and the best performing model is saved.

## Citation

If you use this project in your research or work, please cite it as:

```
@software{vegetable_classifier,
  author = {Your Name},
  title = {Vegetable Classifier: Deep Learning for Vegetable Recognition},
  year = {2025},
  url = {https://github.com/yourusername/vegetable-classifier}
}
```

## License

[MIT License](LICENSE)
