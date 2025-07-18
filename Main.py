import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import numpy as np

# 1. Veri yükleme ve ön işleme
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 2. Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

datagen.fit(X_train)

# 3. Model mimarisi (daha derin ve BatchNorm ile)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding="same", input_shape=(28,28,1)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. EarlyStopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 5. Model eğitimi
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=20,
    callbacks=[early_stop],
    verbose=2
)

# 6. Modeli kaydet
model.save("model.keras")

# 7. Eğitim performansını görselleştir
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.title("Kayıp Grafiği")

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.title("Doğruluk Grafiği")

plt.show()

# 8. İlk 10 örneği göster (daha uygun layout)
plt.figure(figsize=(15,2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i].reshape(28,28), cmap="gray")
    plt.axis("off")
plt.suptitle("İlk 10 Eğitim Görüntüsü")
plt.show()
