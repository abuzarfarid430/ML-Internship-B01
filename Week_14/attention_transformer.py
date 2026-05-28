"""
Task 14.3: Attention Mechanisms & Transformers
Transformer encoder for text classification (IMDB sentiment)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, Model
import warnings
warnings.filterwarnings('ignore')

VOCAB_SIZE  = 10000
MAX_LEN     = 200
EMBED_DIM   = 64
NUM_HEADS   = 4
FF_DIM      = 128
EPOCHS      = 5
BATCH_SIZE  = 64
SAVE_DIR    = "transformer_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── 1. Scaled Dot-Product Attention ─────────────────────────────────────────
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k  = tf.cast(tf.shape(k)[-1], tf.float32)
    scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(d_k)
    if mask is not None:
        scores += (mask * -1e9)
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, v), weights

# ─── 2. Multi-Head Attention Layer ───────────────────────────────────────────
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model   = d_model
        self.depth     = d_model // num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch):
        x = tf.reshape(x, (batch, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, q, k, v, mask=None):
        batch = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch)
        k = self.split_heads(self.wk(k), batch)
        v = self.split_heads(self.wv(v), batch)
        attn_out, weights = scaled_dot_product_attention(q, k, v, mask)
        attn_out = tf.transpose(attn_out, perm=[0,2,1,3])
        attn_out = tf.reshape(attn_out, (batch, -1, self.d_model))
        return self.dense(attn_out), weights

# ─── 3. Positional Encoding ───────────────────────────────────────────────────
def positional_encoding(max_len, d_model):
    positions = np.arange(max_len)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, ...], tf.float32)   # (1, max_len, d_model)

# ─── 4. Transformer Encoder Block ────────────────────────────────────────────
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att   = MultiHeadAttention(d_model, num_heads)
        self.ffn   = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
        ])
        self.ln1   = layers.LayerNormalization(epsilon=1e-6)
        self.ln2   = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn_out, _ = self.att(x, x, x)
        x = self.ln1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x)
        return self.ln2(x + self.drop2(ffn_out, training=training))

# ─── 5. Full Transformer Model ────────────────────────────────────────────────
def build_transformer(vocab_size, max_len, embed_dim, num_heads, ff_dim):
    inp = layers.Input(shape=(max_len,))
    x   = layers.Embedding(vocab_size, embed_dim)(inp)
    pos = positional_encoding(max_len, embed_dim)
    x   = x + pos
    x   = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(0.1)(x)
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name="TransformerClassifier")

# ─── 6. Load IMDB Dataset ─────────────────────────────────────────────────────
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test  = tf.keras.preprocessing.sequence.pad_sequences(x_test,  maxlen=MAX_LEN)
print(f"Train: {x_train.shape}  Test: {x_test.shape}")

# ─── 7. Train Transformer ─────────────────────────────────────────────────────
model = build_transformer(VOCAB_SIZE, MAX_LEN, EMBED_DIM, NUM_HEADS, FF_DIM)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
print("\nTraining Transformer...")
hist = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 validation_split=0.1, verbose=1)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTransformer Test Accuracy: {acc:.4f}")

# ─── 8. LSTM Comparison ───────────────────────────────────────────────────────
lstm_model = tf.keras.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    layers.LSTM(64, return_sequences=True), layers.LSTM(32),
    layers.Dense(64, activation='relu'), layers.Dense(1, activation='sigmoid'),
], name="LSTM")
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nTraining LSTM for comparison...")
lstm_hist = lstm_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                           validation_split=0.1, verbose=1)
lstm_loss, lstm_acc = lstm_model.evaluate(x_test, y_test, verbose=0)
print(f"LSTM Test Accuracy: {lstm_acc:.4f}")

# ─── 9. Visualize Attention Weights ──────────────────────────────────────────
# Extract attention from transformer block
attn_model_inp = model.input
emb_out   = model.layers[1](attn_model_inp)
pos_enc   = positional_encoding(MAX_LEN, EMBED_DIM)
emb_pos   = emb_out + pos_enc
block     = [l for l in model.layers if isinstance(l, TransformerBlock)][0]
_, weights = block.att(emb_pos, emb_pos, emb_pos)

sample = x_test[0:1]
w_val  = tf.keras.backend.function([model.input], [weights])(sample)[0]  # (1,heads,len,len)
avg_w  = w_val[0].mean(axis=0)[:20, :20]                                  # avg heads

plt.figure(figsize=(8, 6))
plt.imshow(avg_w, cmap='viridis', aspect='auto')
plt.colorbar(label='Attention Weight')
plt.title('Multi-Head Attention Weights (first 20 tokens)')
plt.xlabel('Key Position'); plt.ylabel('Query Position')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "attention_weights.png"), dpi=100)
plt.close()

# ─── 10. Training Comparison Plot ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, key, title in zip(axes, ['accuracy','loss'], ['Accuracy','Loss']):
    ax.plot(hist.history[key],           label='Transformer Train')
    ax.plot(hist.history[f'val_{key}'],  label='Transformer Val')
    ax.plot(lstm_hist.history[key],      label='LSTM Train',      linestyle='--')
    ax.plot(lstm_hist.history[f'val_{key}'], label='LSTM Val',    linestyle='--')
    ax.set_title(f'{title} Comparison'); ax.set_xlabel('Epoch')
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "model_comparison.png"), dpi=100)
plt.close()

# ─── 11. Save Model ───────────────────────────────────────────────────────────
model.save(os.path.join(SAVE_DIR, "transformer_classifier.keras"))
print(f"\nTransformer vs LSTM — Transformer: {acc:.4f} | LSTM: {lstm_acc:.4f}")
print(f"Outputs saved to '{SAVE_DIR}/'")
