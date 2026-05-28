"""
Task 14.1: Generative Adversarial Networks (GAN)
DCGAN implementation for MNIST image generation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import warnings
warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
LATENT_DIM   = 100
EPOCHS       = 30
BATCH_SIZE   = 128
IMG_SHAPE    = (28, 28, 1)
SAVE_DIR     = "gan_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── 1. Load Dataset ──────────────────────────────────────────────────────────
print("Loading MNIST dataset...")
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5          # normalize to [-1,1]
x_train = x_train[..., np.newaxis]                               # add channel dim
print(f"Dataset shape: {x_train.shape}")

# ─── 2. Build Generator (DCGAN) ───────────────────────────────────────────────
def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, 256)),

        layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same',
                               use_bias=False, activation='tanh'),
    ], name="Generator")
    return model

# ─── 3. Build Discriminator (DCGAN) ──────────────────────────────────────────
def build_discriminator(img_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=img_shape),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1),
    ], name="Discriminator")
    return model

generator     = build_generator(LATENT_DIM)
discriminator = build_discriminator(IMG_SHAPE)
generator.summary()
discriminator.summary()

# ─── 4. Loss & Optimizers ────────────────────────────────────────────────────
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_optimizer  = optimizers.Adam(1e-4)
disc_optimizer = optimizers.Adam(1e-4)

def discriminator_loss(real_out, fake_out):
    return cross_entropy(tf.ones_like(real_out), real_out) + \
           cross_entropy(tf.zeros_like(fake_out), fake_out)

def generator_loss(fake_out):
    return cross_entropy(tf.ones_like(fake_out), fake_out)

# ─── 5. Training Step ────────────────────────────────────────────────────────
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images  = generator(noise, training=True)
        real_output  = discriminator(images, training=True)
        fake_output  = discriminator(fake_images, training=True)
        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)

    gen_grads  = gen_tape.gradient(g_loss,  generator.trainable_variables)
    disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads,  generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    return g_loss, d_loss

# ─── 6. Save Generated Images ────────────────────────────────────────────────
seed = tf.random.normal([16, LATENT_DIM])

def save_images(epoch):
    preds = generator(seed, training=False)
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(preds[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        ax.axis('off')
    plt.suptitle(f"GAN Generated Images — Epoch {epoch+1}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f"epoch_{epoch+1:03d}.png")
    plt.savefig(path, dpi=100)
    plt.close()
    return path

# ─── 7. Training Loop ────────────────────────────────────────────────────────
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)
g_losses, d_losses = [], []

print("\nStarting DCGAN Training...")
for epoch in range(EPOCHS):
    g_batch, d_batch = [], []
    for batch in dataset:
        g, d = train_step(batch)
        g_batch.append(float(g))
        d_batch.append(float(d))
    g_losses.append(np.mean(g_batch))
    d_losses.append(np.mean(d_batch))
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | G-loss: {g_losses[-1]:.4f} | D-loss: {d_losses[-1]:.4f}")
    if (epoch + 1) % 10 == 0 or epoch == 0:
        save_images(epoch)

# ─── 8. Training Progress Plot ───────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(g_losses, label='Generator Loss', color='blue')
plt.plot(d_losses, label='Discriminator Loss', color='red')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('DCGAN Training Progress')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_loss.png"), dpi=100)
plt.close()

# ─── 9. Final generated grid ─────────────────────────────────────────────────
final_path = save_images(EPOCHS - 1)
print(f"\nTraining complete. Images saved to '{SAVE_DIR}/'")
print(f"Final image grid: {final_path}")
