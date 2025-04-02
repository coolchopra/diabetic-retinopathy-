Diabetic Retinopathy Detection using EfficientNetB0 & APTOS Data
Key Features
Deep Learning Model: EfficientNetB0-based classification of fundus images.

Dataset: APTOS 2019 Blindness Detection dataset.

Preprocessing: Image augmentation and normalization.

Training Strategy: Early stopping and learning rate adjustments.

Evaluation: Confusion matrix and accuracy metrics.

1. Dataset Handling
Uses APTOS 2019 dataset from Kaggle.

CSV files (train.csv, valid.csv, test.csv) contain image names and labels.

Images are classified into five categories:

0 - No Diabetic Retinopathy

1 - Mild

2 - Moderate

3 - Severe

4 - Proliferative

2. Model Architecture
Base Model: EfficientNetB0 (Pretrained on ImageNet).

Modifications:

GlobalAveragePooling2D for feature extraction.

Dense Layers with ReLU activation.

Dropout Layers for regularization.

Softmax Activation for multi-class classification.

3. Data Preprocessing & Augmentation
Image Resizing: 224x224 pixels.

Image Augmentation using ImageDataGenerator:

Rotation, zoom, width/height shift, horizontal flip.

Normalization: Pixel values scaled for faster training.

4. Training Strategy
Optimizer: Adam optimizer.

Loss Function: Categorical Crossentropy.

Callbacks Used:

EarlyStopping (Stops training if validation loss does not improve).

ReduceLROnPlateau (Reduces learning rate on plateau).

TensorBoard for visualization.

Batch Size: 32

Epochs: 50+ (adjusted with early stopping).

5. Model Evaluation
Confusion Matrix for assessing classification accuracy.

Performance Metrics:

Accuracy

Precision, Recall, F1-score

ROC-AUC Score

Visualization: Uses matplotlib and seaborn for plotting.

6. Deployment & Usage
Model Export: Saved in HDF5 (.h5) format.

Future Integration: Can be deployed via Flask/Django API for real-world usage.

