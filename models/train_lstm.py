import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Load data
X = np.load("dataset/X.npy")
y = np.load("dataset/y.npy")

# Fix data types (this is the key fix!)
X = X.astype(np.float32)
y = y.astype(np.float32)

# Print shapes for debugging
print(f"[INFO] X shape: {X.shape}, dtype: {X.dtype}")
print(f"[INFO] y shape: {y.shape}, dtype: {y.dtype}")

# Reshape X if needed (LSTM expects 3D input: [samples, timesteps, features])
if len(X.shape) == 2:
    X = X.reshape((X.shape[0], X.shape[1], 1))
    print(f"[INFO] Reshaped X to: {X.shape}")

# Build model with proper Input layer
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),  # Better than input_shape in LSTM
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# Train model
history = model.fit(X, y, 
                    epochs=10, 
                    batch_size=16, 
                    validation_split=0.2,
                    verbose=1)

# Save model
model.save("models/lstm_model.h5")
print("[INFO] LSTM predictive model trained and saved!")
print(f"[INFO] Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"[INFO] Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")