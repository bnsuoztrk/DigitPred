import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Görsel yolu
image_path = "Deneme.png"
model_path = "model.keras"

# Görseli gri tonlamalı oku
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

# Orijinal görseli 280x280 boyutuna getir (görselleştirme için)
original_img = cv2.resize(img, (280, 280))

# Model için 28x28 boyutuna küçült
img_resized = cv2.resize(img, (28, 28))

# Renkleri tersle (model siyah zemin + beyaz rakam bekliyor)
img_inverted = 255 - img_resized

# Normalize et (0-1 aralığına)
img_norm = img_inverted.astype("float32") / 255.0

# Modelin beklediği shape: (1, 28, 28, 1)
img_input = img_norm.reshape(1, 28, 28, 1)

# Modeli yükle
model = tf.keras.models.load_model(model_path)

# Tahmin yap
predictions = model.predict(img_input)
predicted_label = np.argmax(predictions)

# Sonucu göster
plt.figure(figsize=(4,4))
plt.imshow(original_img, cmap="gray")
plt.title(f"Tahmin Edilen Rakam: {predicted_label}", fontsize=16)
plt.axis("off")
plt.show()

print(f"Modelin tahmini: {predicted_label}")
