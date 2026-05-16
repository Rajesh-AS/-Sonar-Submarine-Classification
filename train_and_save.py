import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("Loading dataset...")
dataset = pd.read_csv('sonar_dataset.csv', header=None)

# Preprocessing
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode categorical target variable (R, M) -> (0, 1)
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save the scaler using pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)
print("Saved StandardScaler to scaler.pkl")

# Build Model with Dropout Regularization
print("Building the Neural Network model with Dropout...")
model = Sequential()
model.add(Dense(units=60, activation='relu', input_dim=60))
model.add(Dropout(0.5))
model.add(Dense(units=30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the model
model.save('sonar_model.keras')
print("Saved Model to sonar_model.keras")
