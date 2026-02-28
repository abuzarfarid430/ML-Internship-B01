import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Normalize pixel values (0-255 → 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# CNN expects: (samples, height, width, channels)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(X_train)

# 4. Build CNN Architecture


model = models.Sequential()

# First Convolution Block
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))

# Second Convolution Block
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# Flatten
model.add(layers.Flatten())

# Fully Connected Layer
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

# Output Layer (10 classes)
model.add(layers.Dense(10, activation='softmax'))

model.summary()


# 5. Compile Model


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train Model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=10,
    validation_data=(X_test, y_test)
)


test_loss, test_acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", test_acc)


plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy vs Epochs")
plt.show()


filters, biases = model.layers[0].get_weights()

plt.figure(figsize=(8,8))
for i in range(6):
    f = filters[:, :, 0, i]
    plt.subplot(3, 3, i+1)
    plt.imshow(f, cmap='gray')
    plt.axis('off')
plt.suptitle("First Layer Filters")
plt.show()


def visualize_feature_maps(model, image):
    """
    Visualizes feature maps of first convolution layer.
    """
    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[0].output
    )
    
    feature_maps = feature_model.predict(image.reshape(1,28,28,1))
    
    plt.figure(figsize=(10,6))
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(feature_maps[0,:,:,i], cmap='gray')
        plt.axis('off')
    plt.suptitle("Feature Maps")
    plt.show()


# Visualize for first test image
visualize_feature_maps(model, X_test[0])


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

#convert to tflite

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("cnn_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted and saved as cnn_model.tflite")