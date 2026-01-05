import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("\n[LOADING DATA AND MODELS]")
print("="*50)

# Load LSTM data (sequences: shape = (samples, 30, 1))
X_lstm = np.load("dataset/X.npy").astype(np.float32)
y = np.load("dataset/y.npy").astype(np.float32)

print(f"Original LSTM data shape: {X_lstm.shape}")
print(f"Labels shape: {y.shape}")

# Convert LSTM sequences to ML features
# Extract meaningful statistics from the 30-frame sequences
X_ml = np.column_stack([
    X_lstm.mean(axis=1).flatten(),      # Average EAR over 30 frames
    X_lstm.std(axis=1).flatten(),       # Variability in EAR
])

print(f"Converted ML data shape: {X_ml.shape}")

# Load models
ml_model = joblib.load("models/ml_model.pkl")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")

# Create train/test split (80/20)
split_idx = int(0.8 * len(y))

X_ml_test = X_ml[split_idx:]
X_lstm_test = X_lstm[split_idx:]
y_test = y[split_idx:]

print(f"Test set size: {len(y_test)} samples")


# EVALUATE ML MODEL
print("\n[ML MODEL EVALUATION]")
print("="*50)

y_ml_pred = ml_model.predict(X_ml_test)
ml_accuracy = accuracy_score(y_test, y_ml_pred)

print(f"Accuracy: {ml_accuracy:.4f} ({ml_accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_ml_pred, 
                          target_names=['Alert', 'Drowsy'],
                          digits=4))

cm_ml = confusion_matrix(y_test, y_ml_pred)
print("\nConfusion Matrix:")
print(f"              Predicted")
print(f"              Alert  Drowsy")
print(f"Actual Alert   {cm_ml[0][0]:5d}  {cm_ml[0][1]:5d}")
print(f"       Drowsy  {cm_ml[1][0]:5d}  {cm_ml[1][1]:5d}")


# EVALUATE LSTM MODEL
print("\n[LSTM MODEL EVALUATION]")
print("="*50)

# Get predictions
y_lstm_pred_prob = lstm_model.predict(X_lstm_test, verbose=0)
y_lstm_pred = (y_lstm_pred_prob > 0.5).astype(int).flatten()

lstm_accuracy = accuracy_score(y_test, y_lstm_pred)

print(f"Accuracy: {lstm_accuracy:.4f} ({lstm_accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_lstm_pred,
                          target_names=['Alert', 'Drowsy'],
                          digits=4))

cm_lstm = confusion_matrix(y_test, y_lstm_pred)
print("\nConfusion Matrix:")
print(f"              Predicted")
print(f"              Alert  Drowsy")
print(f"Actual Alert   {cm_lstm[0][0]:5d}  {cm_lstm[0][1]:5d}")
print(f"       Drowsy  {cm_lstm[1][0]:5d}  {cm_lstm[1][1]:5d}")


# MODEL COMPARISON
print("\n[MODEL COMPARISON SUMMARY]")
print("="*50)
print(f"{'Model':<15} {'Accuracy':<12} {'Performance'}")
print("-"*50)
print(f"{'ML Baseline':<15} {ml_accuracy:.4f} ({ml_accuracy*100:5.2f}%)")
print(f"{'LSTM':<15} {lstm_accuracy:.4f} ({lstm_accuracy*100:5.2f}%)")
print("-"*50)

if lstm_accuracy > ml_accuracy:
    improvement = (lstm_accuracy - ml_accuracy) * 100
    print(f"\n✓ LSTM is BETTER by {improvement:.2f}%")
    print("  → LSTM captures temporal patterns in blink sequences")
elif ml_accuracy > lstm_accuracy:
    improvement = (ml_accuracy - lstm_accuracy) * 100
    print(f"\n✓ ML Baseline is BETTER by {improvement:.2f}%")
    print("  → Simple features may be sufficient for this task")
else:
    print(f"\n= Models perform EQUALLY well")

# False positive/negative analysis
print("\n[ERROR ANALYSIS]")
print("="*50)
ml_fp = cm_ml[0][1]  # Alert predicted as Drowsy
ml_fn = cm_ml[1][0]  # Drowsy predicted as Alert
lstm_fp = cm_lstm[0][1]
lstm_fn = cm_lstm[1][0]

print(f"{'Model':<15} {'False Positives':<20} {'False Negatives'}")
print("-"*50)
print(f"{'ML Baseline':<15} {ml_fp:<20} {ml_fn}")
print(f"{'LSTM':<15} {lstm_fp:<20} {lstm_fn}")
print("\nNote: False Negatives (missing drowsiness) are MORE dangerous!")

print("\n[EVALUATION COMPLETE]")