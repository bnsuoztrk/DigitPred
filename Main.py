import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    brightness_range=[0.8,1.2],
    fill_mode='nearest'
)
datagen.fit(X_train)

# 3. Model mimarisi 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding="same", input_shape=(28,28,1)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(32, (3,3), padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(64, (3,3), padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

#Model eğitimi
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=5,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# Modeli kaydet
model.save("model.keras")

#Eğitim performansını görselleştir
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
