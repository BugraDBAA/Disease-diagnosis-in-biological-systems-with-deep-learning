
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np

image_size = (180,180)
batch_size = 32
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
   # x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128,256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=3)

keras.utils.plot_model(model, show_shapes=True)


model.load_weights("save_at_51.h5")
img1 = keras.preprocessing.image.load_img(
    "indir.jpg", target_size=image_size
)

img_array1 = keras.preprocessing.image.img_to_array(img1)
img_array1 = tf.expand_dims(img_array1, 0)  # Create batch axis
predictions1 = model.predict(img_array1)
maximum1=np.argmax(predictions1)

if(maximum1==0):
    print("Elma küllemesi hastalığı")
    print("Kültürel Önlemler")
    print("Küllemden zarar görmüş sürgünler kış budamasıyla, hastalıklı kısmın 15 cm altından kesilip bajçeden uzaklaştırılmalıdır.")
    print("Kış budaması sırasında gözden kaçan ve ilkbaharda tepe tomurcukları küllemeli olarak gelişen sürgünler ile küllemeli olarak gelişen yaprak ve çiçek demetleri toplanmalı ve bahçeden uzaklaştırılmalıdır.")
    print("Kimyasal Mücadeleler")
    print("1.İlaçlama: Pembe çiçek tomurcuğu döneminde yapılır.")
    print("2.İlaçlama: Çiçek taç yapraklarının %60-70'i döküldüğünde yapılır.")
elif (maximum1==1):
    print("Elma karaleke hastalığı")
    print("Kültürel Önlemler")
    print("Primer enfeksiyon kaynağı olan yere dökülmüş lekeli yapraklar sonbaharda toplanıp yakılmalı veya gömülmelidir")
    print("Sıracalı dallar budanıp bahçeden uzaklaştırılmalıdır")
    print("Ağaçlar, yapraklardaki nemin daha hızlı kuruyabilmesi için hava akımına izin verecek şekilde taçlandırılmalı ve uygun aralıklar ile dikilmelidir.")
    print("Kimyasal Mücadeleler")
    print("1.İlaçlama: Çiçek gözleri kabardığında yapılır.")
    print("2.İlaçlama: Pembe çiçek tomurcuğu döneminde yapılır.")
    print("3.İlaçlama: Çiçek taç yaprakları %70-80 döküldüğünde yapılır.")
elif (maximum1==2):
    print("Sağlıklı")
