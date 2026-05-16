# 🌊 Sonar Submarine Classification

This project uses a Deep Learning Artificial Neural Network (ANN) to classify sonar signals as either a **Mine** (💣) or a **Rock** (🪨) based on frequency readings. 

## 📋 Project Overview
The dataset contains 60 different sonar frequency measurements. The objective of this project is to build an intelligent model capable of predicting the type of object the sonar has detected. The Neural Network uses **Dropout Regularization (50%)** to prevent the model from overfitting to the training data.

A fully functional and responsive web interface is included, built with **Streamlit**, to provide real-time classifications.

## 🛠️ Tech Stack
- **Python 3.x**
- **TensorFlow / Keras** (Deep Learning & Neural Networks)
- **Scikit-Learn** (Data Preprocessing, Scaling, Label Encoding)
- **Streamlit** (Interactive Web App)
- **Pandas & NumPy** (Data Manipulation)

## 📂 Project Structure
- `app.py`: The Streamlit web application. Provides a dark-themed user interface to input 60 frequency values and get a prediction.
- `train.py`: Script to train the Neural Network and evaluate its performance with accuracy & loss plots.
- `train_and_save.py`: Script to train the model and generate the saved artifacts (`sonar_model.keras` & `scaler.pkl`).
- `sonar_model.keras`: The pre-trained Artificial Neural Network model.
- `scaler.pkl`: The StandardScaler object used to normalize the input data during inference.
- `sonar_dataset.csv`: The dataset containing 60 numeric features and the target label (R/M).
- `Model.ipynb`: Jupyter notebook containing data exploration and prototyping.

## 🚀 How to Run Locally

### 1. Install Dependencies
Ensure you have Python installed, then install the required packages:
```bash
pip install tensorflow scikit-learn pandas numpy streamlit matplotlib
```

### 2. (Optional) Re-train the Model
If you want to train the model yourself and regenerate the `.keras` and `.pkl` files:
```bash
python train_and_save.py
```

### 3. Start the Web App
Run the following command to start the Streamlit application:
```bash
streamlit run app.py
# OR
python -m streamlit run app.py
```

This will automatically open your default web browser to the application page.

## 🎯 Example Data (Rock)
You can test the app using the following sample data (which should classify as a Rock):
```
0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032
```
