"""
Task 14.4: Time Series Forecasting with LSTM
Stock price forecasting using LSTM vs ARIMA
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

SAVE_DIR  = "lstm_output"
os.makedirs(SAVE_DIR, exist_ok=True)
SEQ_LEN   = 60
EPOCHS    = 30
BATCH     = 32

# ─── 1. Generate Synthetic Stock-Price Data ────────────────────────────────
np.random.seed(42)
t    = np.linspace(0, 10*np.pi, 1000)
data = (np.sin(t) + np.sin(0.5*t) + 0.05*np.random.randn(len(t))) * 50 + 200
print(f"Dataset size: {len(data)} time steps")

# ─── 2 & 3. Normalize ─────────────────────────────────────────────────────────
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data.reshape(-1,1))

# ─── 2. Sliding Window ────────────────────────────────────────────────────────
def make_sequences(arr, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled, SEQ_LEN)

# ─── 4. Temporal Split ────────────────────────────────────────────────────────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ─── 5. LSTM Model ────────────────────────────────────────────────────────────
model = tf.keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
], name="LSTM_Forecaster")
model.compile(optimizer='adam', loss='mse')
model.summary()

# ─── 6. Train with Early Stopping ────────────────────────────────────────────
cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
print("\nTraining LSTM...")
hist = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
                 validation_split=0.1, callbacks=[cb], verbose=1)

# ─── 7. Predictions ───────────────────────────────────────────────────────────
y_pred_scaled = model.predict(X_test)

# ─── 8. Inverse Transform ─────────────────────────────────────────────────────
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test.reshape(-1,1))

# ─── 9. RMSE & MAE ────────────────────────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
print(f"\nLSTM RMSE: {rmse:.4f}  MAE: {mae:.4f}")

# ─── 10. ARIMA Comparison ─────────────────────────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA
    train_series = data[:split + SEQ_LEN]
    arima = ARIMA(train_series, order=(5,1,0)).fit()
    arima_pred = arima.forecast(steps=len(y_true))
    arima_rmse = np.sqrt(mean_squared_error(y_true, arima_pred))
    arima_mae  = mean_absolute_error(y_true, arima_pred)
    print(f"ARIMA RMSE: {arima_rmse:.4f}  MAE: {arima_mae:.4f}")
    has_arima = True
except Exception as e:
    print(f"ARIMA skipped: {e}")
    has_arima = False

# ─── 11. Plot Actual vs Predicted ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(y_true,  label='Actual',         color='black')
axes[0].plot(y_pred,  label='LSTM Predicted', color='blue', alpha=0.8)
if has_arima:
    axes[0].plot(arima_pred, label='ARIMA', color='red', linestyle='--', alpha=0.7)
axes[0].set_title('Time Series Forecasting: Actual vs Predicted')
axes[0].set_xlabel('Time Step'); axes[0].set_ylabel('Value')
axes[0].legend()

axes[1].plot(hist.history['loss'],     label='Train Loss')
axes[1].plot(hist.history['val_loss'], label='Val Loss')
axes[1].set_title('LSTM Training Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "forecast_results.png"), dpi=100)
plt.close()

# Metrics comparison bar chart
models  = ['LSTM']
rmses   = [rmse]
maes    = [mae]
if has_arima:
    models.append('ARIMA')
    rmses.append(arima_rmse)
    maes.append(arima_mae)

x = np.arange(len(models))
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].bar(x, rmses, color=['blue','red'][:len(models)])
axes[0].set_title('RMSE Comparison'); axes[0].set_xticks(x); axes[0].set_xticklabels(models)
axes[1].bar(x, maes, color=['blue','red'][:len(models)])
axes[1].set_title('MAE Comparison');  axes[1].set_xticks(x); axes[1].set_xticklabels(models)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "metrics_comparison.png"), dpi=100)
plt.close()

# ─── 12. Save Best Model ──────────────────────────────────────────────────────
model.save(os.path.join(SAVE_DIR, "best_lstm_model.keras"))
print(f"\nOutputs saved to '{SAVE_DIR}/'")
