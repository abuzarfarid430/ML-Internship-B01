"""
Task 14.2: Variational Autoencoder (VAE)
VAE for image generation and reconstruction on MNIST
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

LATENT_DIM = 2
EPOCHS     = 20
BATCH_SIZE = 128
SAVE_DIR   = "vae_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── 1. Load Dataset ──────────────────────────────────────────────────────────
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = x_train[..., np.newaxis]
x_test  = x_test[..., np.newaxis]
print(f"Train: {x_train.shape}  Test: {x_test.shape}")

# ─── 2. Sampling Layer (Reparameterization) ────────────────────────────────
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

# ─── 3. Encoder ───────────────────────────────────────────────────────────────
def build_encoder(latent_dim):
    inp = layers.Input(shape=(28,28,1))
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inp)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    z_mean    = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z         = Sampling()([z_mean, z_log_var])
    return Model(inp, [z_mean, z_log_var, z], name='Encoder')

# ─── 4. Decoder ───────────────────────────────────────────────────────────────
def build_decoder(latent_dim):
    inp = layers.Input(shape=(latent_dim,))
    x = layers.Dense(7*7*64, activation='relu')(inp)
    x = layers.Reshape((7,7,64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    out = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    return Model(inp, out, name='Decoder')

encoder = build_encoder(LATENT_DIM)
decoder = build_decoder(LATENT_DIM)
encoder.summary(); decoder.summary()

# ─── 5. VAE Model ─────────────────────────────────────────────────────────────
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder, self.decoder = encoder, decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.recon_loss_tracker  = tf.keras.metrics.Mean(name='recon_loss')
        self.kl_loss_tracker     = tf.keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            recon = self.decoder(z)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, recon), axis=(1,2))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total = recon_loss + kl_loss
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

vae = VAE(encoder, decoder)
vae.compile(optimizer='adam')
print("\nTraining VAE...")
history = vae.fit(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# ─── 9. Visualize Original vs Reconstructed ──────────────────────────────────
n = 10
samples = x_test[:n]
_, _, z = encoder(samples)
recons  = decoder(z).numpy()

fig, axes = plt.subplots(2, n, figsize=(15, 3))
for i in range(n):
    axes[0,i].imshow(samples[i,:,:,0], cmap='gray'); axes[0,i].axis('off')
    axes[1,i].imshow(recons[i,:,:,0],  cmap='gray'); axes[1,i].axis('off')
axes[0,0].set_ylabel("Original",      fontsize=10)
axes[1,0].set_ylabel("Reconstructed", fontsize=10)
plt.suptitle("VAE: Original vs Reconstructed", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "reconstruction.png"), dpi=100)
plt.close()

# ─── 10. Latent Space Visualization ──────────────────────────────────────────
z_mean, _, _ = encoder(x_test[:3000])
plt.figure(figsize=(7, 6))
sc = plt.scatter(z_mean[:,0].numpy(), z_mean[:,1].numpy(),
                 c=y_test[:3000], cmap='tab10', alpha=0.6, s=8)
plt.colorbar(sc, label='Digit Class')
plt.xlabel('z[0]'); plt.ylabel('z[1]')
plt.title('VAE Latent Space (2D)')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "latent_space.png"), dpi=100)
plt.close()

# ─── 11. Generate New Images ──────────────────────────────────────────────────
z_sample = np.random.normal(0, 1, (16, LATENT_DIM)).astype("float32")
gen_imgs  = decoder(z_sample).numpy()
fig, axes = plt.subplots(4, 4, figsize=(6,6))
for i, ax in enumerate(axes.flat):
    ax.imshow(gen_imgs[i,:,:,0], cmap='gray'); ax.axis('off')
plt.suptitle("VAE Generated Images", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "generated.png"), dpi=100)
plt.close()

# ─── 12. Save Model ───────────────────────────────────────────────────────────
encoder.save(os.path.join(SAVE_DIR, "vae_encoder.keras"))
decoder.save(os.path.join(SAVE_DIR, "vae_decoder.keras"))
print(f"\nVAE training complete. Outputs saved to '{SAVE_DIR}/'")
