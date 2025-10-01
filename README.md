🩺 Face Mask Detection using CNN

📌 Project Overview

This project aims to build a Convolutional Neural Network (CNN) model that can detect whether a person is wearing a face mask or not.
It uses an image dataset with two classes:

With Mask 😷

Without Mask 🙂

The model was trained and tested on Kaggle using TensorFlow/Keras.

📂 Dataset

Dataset Source: Face Mask Detection Dataset (Kaggle)

Dataset Structure:

dataset/
  ├── with_mask/
  └── without_mask/


Total Images: ~3,800

Training: ~3,067 images

Validation: ~766 images

⚙️ Technologies Used

Python 🐍

TensorFlow / Keras (for deep learning)

Matplotlib & Seaborn (for visualization)

NumPy & Pandas (data handling)

Kaggle Notebook (for training & testing)

🧠 Model Architecture

Input Layer: 128x128 RGB images

Convolution + MaxPooling layers

Dropout layers for regularization

Dense layers with ReLU activation

Output Layer: Sigmoid activation (binary classification)

📊 Training Results

Epochs: 10

Optimizer: Adam

Loss Function: Binary Crossentropy

Final Accuracy: ~95% on validation data

🚀 How to Run

Open the notebook in Kaggle.

Upload the dataset under Add Input → Face Mask Dataset.

Run all cells to:

Preprocess the dataset

Train the CNN model

Test predictions on custom images

🖼️ Predictions

Example predictions:

Input: Person with mask → Predicted: With Mask (0.98 confidence)

Input: Person without mask → Predicted: Without Mask (0.95 confidence)

✅ Future Improvements

Add data augmentation for better generalization.

Try Transfer Learning (ResNet, MobileNet, VGG16) for higher accuracy.

Deploy model as a web app (Flask/Streamlit).
