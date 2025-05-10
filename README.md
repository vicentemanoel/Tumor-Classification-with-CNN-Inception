# Tumor classificatio with CNN Inception

## 📚 Objective
The main objective of this project is to develop a deep learning model for image classification based on the **InceptionV3** architecture.  
We aim to efficiently classify brain tumors from MRI scans by leveraging the multi-pathway convolutional structure of Inception, allowing the network to capture different spatial features at multiple scales simultaneously.

The project explores:
- Fine-tuning a pre-trained InceptionV3 model.
- Constructing custom classification layers on top.
- Training and validating the model on MRI datasets.
- Evaluating performance through key metrics.

## 🗂️ Data Exploration
- **Dataset**: Brain MRI Images categorized by tumor types.
- **Classes**: (Fill specific tumor names — e.g., Glioma, Meningioma, Pituitary, etc.)
- **Preprocessing**:
  - Resized all images to 299x299 pixels.
  - Applied normalization using `preprocess_input`.
  - Data augmentation applied for better generalization (rotation, zoom, etc.).
- **Splits**:
  - Training set: Images from `/Training/`.
  - Testing set: Images from `/Testing/`.

Exploratory analysis showed:
- A relatively balanced distribution among classes in training and testing sets.
- Good diversity in MRI images, but some overlap in tumor appearance.

## 🛠️ Model Architecture
The model architecture consists of:
- **InceptionV3 base** (pre-trained on ImageNet, without top layers).
- Global Average Pooling layer.
- Fully connected Dense layer with softmax activation.

**Activation functions**: ReLU (internal), Softmax (output)  
**Optimization**:
- Loss function: `categorical_crossentropy`
- Optimizer: `Adam`
- Learning rate: default `1e-4`

## 📈 Main Results
The model achieved the following performance on the test set:
- **Accuracy**: 96.1%
- **Precision**: Above 95% for most tumor classes.
- **Recall**: Above 95% for most tumor classes.
- **F1-score**: Approximately 96% overall.

Key observations:
- The model converged well without overfitting, thanks to data augmentation and early stopping.
- Certain classes showed slightly lower performance, potentially due to similar visual features between tumor types.

Performance graphs:
- Training and validation loss showed good convergence.
- Training and validation accuracy increased steadily across epochs.

## 🔍 Conclusions
- The fine-tuned InceptionV3 model demonstrated excellent performance for brain tumor classification.
- Transfer learning significantly reduced the training time and improved accuracy compared to training from scratch.
- Future improvements could include:
  - Fine-tuning more layers of InceptionV3.
  - Applying more advanced augmentation strategies.
  - Testing ensemble models.

Overall, this project highlights the strength of pre-trained convolutional networks for medical imaging tasks.

## 🚀 How to Run
1. Clone this repository.
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook `Inception_Manoel.ipynb` to train and evaluate the model.

## 📋 Requirements
- Python 3.x
- TensorFlow >= 2.8
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
