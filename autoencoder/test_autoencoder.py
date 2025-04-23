import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

# Fungsi untuk load images dari dua folder (input dan target)
def load_images_from_folder(folderA, folderB, img_size=(128, 128)):
    image_paths_A = sorted([os.path.join(folderA, f) for f in os.listdir(folderA) if f.endswith(('jpg', 'png'))])
    image_paths_B = sorted([os.path.join(folderB, f) for f in os.listdir(folderB) if f.endswith(('jpg', 'png'))])

    images_A, images_B = [], []
    for path_A, path_B in zip(image_paths_A, image_paths_B):
        try:
            img_A = load_img(path_A, target_size=img_size)
            img_B = load_img(path_B, target_size=img_size)
            images_A.append(img_to_array(img_A) / 255.0)
            images_B.append(img_to_array(img_B) / 255.0)
        except Exception as e:
            print(f"Error loading image {path_A} or {path_B}: {e}")

    print(f"Loaded {len(images_A)} image pairs from {folderA} & {folderB}")
    return np.array(images_A), np.array(images_B)

# Load training dan testing data
x_train, y_train = load_images_from_folder("dataset/trainA", "dataset/trainB")
x_test, y_test = load_images_from_folder("dataset/testA", "dataset/testB")

# Model U-Net sederhana
def build_unet(input_shape=(128, 128, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)

    # Decoder
    u2 = UpSampling2D((2, 2))(b)
    concat2 = concatenate([u2, c2])
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat2)

    u1 = UpSampling2D((2, 2))(c3)
    concat1 = concatenate([u1, c1])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c4)

    return Model(inputs, outputs)

# Callback untuk print loss manual tiap epoch
class LossPrinter(Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

# Bangun dan compile model
model = build_unet()
model.compile(optimizer='adam', loss='mean_squared_error')

# Training tanpa progress bar
model.fit(
    x_train, y_train,
    epochs=40,
    batch_size=8,
    validation_split=0.1,
    shuffle=True,
    verbose=0,
    callbacks=[LossPrinter()]
)

# Simpan model
model.save("unet_autoencoder_model.h5")

# Predict untuk testing data
predicted_imgs = model.predict(x_test)

# Visualisasi hasil prediksi vs target
for i in range(min(5, len(x_test))):
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(x_test[i])
    plt.title("Input (testA)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(y_test[i])
    plt.title("Target (testB)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_imgs[i])
    plt.title("Output (Prediksi)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()