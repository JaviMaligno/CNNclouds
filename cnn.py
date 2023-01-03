# %%
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
cd = os.getcwd()
cloud_dir = os.path.join(cd, 'clouds')
train = tf.keras.utils.image_dataset_from_directory(cloud_dir, labels = "inferred", label_mode="categorical", batch_size=1)

# https://keras.io/examples/vision/image_classification_from_scratch/

## This code is for batch_size = 1. The model needs to set a batch size
plt.figure(figsize=(10, 10))
for j, (images, labels) in enumerate(train.take(9)):
    #print(labels)
    ax = plt.subplot(3, 3, j + 1)
    plt.imshow(images[0].numpy().astype("uint8"))
    plt.title(int(tf.where(tf.equal(labels[0], 1))))
    plt.axis("off")
""" 
The code below works for batch_size = None

plt.figure(figsize=(10, 10))
for j, (images, labels) in enumerate(train.take(9)):
    #print(labels)
    ax = plt.subplot(3, 3, j + 1)
    plt.imshow(images.numpy().astype("uint8"))
    plt.title(int(tf.where(tf.equal(labels, 1))))
    plt.axis("off") """

# %%
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
    ]
)
plt.figure(figsize=(10, 10))
for images, _ in train.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images[0]) #I can apply augmentation to the whole batch and then select [0] in the imshow instead, but that outputs warnings
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images.numpy().astype("uint8")) 
        plt.axis("off")

# %%
# Apply `data_augmentation` to the training images.

train = train.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)


# Prefetching samples in GPU memory helps maximize GPU utilization.
train = train.prefetch(tf.data.AUTOTUNE)


# %%
# transform the base64 image from the json cloud into a tensor image
from PIL import Image
import base64
import io
import numpy as np
import json

def tensor_image(json_cloud):
    cloud_path = os.path.join(cd, json_cloud)
    with open(cloud_path) as f:
        data = json.load(f)
        image_data = data["instances"][0]["content"]
        base64_decoded = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(base64_decoded))
        image_np = np.array(image)
        tensor_image = tf.keras.utils.array_to_img(
            image_np, data_format=None, scale=True, dtype=None)
    return tensor_image

cloud1 = tensor_image("CLOUD1-JSON")
cloud2 = tensor_image("CLOUD2-JSON")
cloud1

# %%
cloud2

# %%
# Input shape
print(train.element_spec)
shape = train.element_spec[0].shape 
print(*shape, tuple(shape))
train
""" 
Only valid for batch_size = None

plt.figure(figsize=(10, 10))
for j, (images, labels) in enumerate(train.take(9)):
    #print(labels)
    ax = plt.subplot(3, 3, j + 1)
    plt.imshow(images.numpy().astype("uint8"))
    plt.title(int(tf.where(tf.equal(labels[0], 1))))
    plt.axis("off") """

# %%
# Model

from keras import layers
from keras.utils.vis_utils import plot_model
import pydot, pydot_ng, pydotplus
import pydotplus
import graphviz
#https://www.datacamp.com/tutorial/convolutional-neural-networks-python
#https://keras.io/examples/vision/image_classification_from_scratch/
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs) # RGB is in [0,255], this rescales to [0,1] which is better
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    for size in [ 128,256, 512]:
        x = layers.Activation("relu")(x)
        #x = layers.Conv2D(size, (2,2), strides=2, padding="same", input_shape = input_shape)(x) 
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x) # I may not need this as I didn't use batch (or batch =1 I guess)
        
        x = layers.Activation("relu")(x)
        #x = layers.Conv2D(size, (2,2), strides=2, padding="same", input_shape = input_shape)(x) 
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x) 

        x = layers.MaxPooling2D(pool_size=(2, 2),padding='same')(x)
    
    #x = layers.Conv2D(256, (2,2), strides=2, padding="same", input_shape = input_shape)(x) 
    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x) 

    x = layers.Activation("relu")(x)

    activation = "softmax"
    units = num_classes
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    #outputs = layers.Flatten()(outputs)

    return keras.Model(inputs, outputs)

model = make_model(input_shape=(256,256,3), num_classes=3)
tf.keras.utils.plot_model(model, show_shapes=True)

# %%
# Compile and train model

epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train,
    epochs=epochs,
    callbacks=callbacks
) 

# %%
train

# %%
cloud_path_1 = "cloud1.png"
cloud_path_2 = "cloud2.png"

def predict_cloud(image_path):
    img = keras.preprocessing.image.image_utils.load_img( #load_img is inside image_utils
        image_path, target_size=(256,256)
    )
    img_array = keras.preprocessing.image.image_utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]
    cloud_label = {0 : 'cirrus', 1: 'cumulonimbus', 2: 'cumulus'}
    print(f"This image is {100 * score[0]:.2f}% cirrus, {100 * score[1]:.2f}% cumulonimbus, and {100 * score[2]:.2f}% cumulus.")
    predicted = cloud_label[np.argmax(predictions)]

    return predicted

# %%
print('Cloud 1 predicted as ', predict_cloud(cloud_path_1))
print('Cloud 2 predicted as ', predict_cloud(cloud_path_2))
print(predict_cloud(r"clouds\cumulonimbus\1.jpg"))
print(predict_cloud(r"clouds\cirrus\1.jpg"))
print(predict_cloud(r"clouds\cumulus\1.jpg"))

# %%
correct = 0
for i in range(1,21):
    predicted = predict_cloud(rf"clouds\cirrus\{i}.jpg")
    if predicted == 'cirrus':
        correct += 1
    accuracy = f"{correct/20 * 100: .2f}"
    print(i, predicted)
print(f"The accuracy on cirrus is {accuracy} %")

# %%



