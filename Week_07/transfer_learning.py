import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory


IMG_SIZE = (160, 160)
BATCH_SIZE = 32

train_dir = "dataset/train"
val_dir = "dataset/validation"

train_ds = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

#Load MobileNetV2 as base model
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)


base_model.trainable = False


inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_transfer = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)


base_model.trainable = True

# Freeze first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)


scratch_model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(160,160,3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

scratch_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_scratch = scratch_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

transfer_acc = history_transfer.history['val_accuracy'][-1]
finetune_acc = history_finetune.history['val_accuracy'][-1]
scratch_acc = history_scratch.history['val_accuracy'][-1]

comparison = pd.DataFrame({
    "Model": ["Transfer Learning (Frozen)",
              "Transfer Learning (Fine-tuned)",
              "Training From Scratch"],
    "Validation Accuracy": [
        transfer_acc,
        finetune_acc,
        scratch_acc
    ]
})

print("\nPerformance Comparison:")
print(comparison)


# Save as .h5
model.save("transfer_model.h5")

# Save as SavedModel
model.save("transfer_saved_model")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("transfer_model.tflite", "wb") as f:
    f.write(tflite_model)

print("\nModel saved in .h5, SavedModel, and .tflite formats.")

#plot comparison
plt.figure()
plt.bar(
    ["Transfer", "Fine-tune", "Scratch"],
    [transfer_acc, finetune_acc, scratch_acc]
)
plt.title("Model Comparison")
plt.ylabel("Validation Accuracy")
plt.show()