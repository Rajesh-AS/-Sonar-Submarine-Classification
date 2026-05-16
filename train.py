import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Build Model with Dropout Regularization
print("Building the Neural Network model with Dropout...")
model = Sequential()
model.add(Dense(units=60, activation='relu', input_dim=60))
model.add(Dropout(0.5)) # 50% probability to drop neurons
model.add(Dense(units=30, activation='relu'))
model.add(Dropout(0.5)) # 50% probability to drop neurons
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {accuracy * 100:.2f}%')

# Plot training history
print("Plotting the history curves... Close the plot window to exit.")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
