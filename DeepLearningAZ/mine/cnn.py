import pathlib
import time

import keras
import numpy as np
from PIL import Image
import os

# %%

batch_size = 32
img_h = 64
img_w = 64

# %%

training_data_dir = pathlib.Path(
    "./dataset/Part 2 - Convolutional Neural Networks (CNN)/dataset/training_set"
)
training_data = keras.utils.image_dataset_from_directory(
    training_data_dir, image_size=(img_h, img_w), batch_size=batch_size
)
print(training_data.class_names)  # pyright: ignore[reportAttributeAccessIssue]
class_names = training_data.class_names  # pyright: ignore[reportAttributeAccessIssue]
num_classes = len(training_data.class_names)  # pyright: ignore[reportAttributeAccessIssue]


data_aug = keras.Sequential(
    [
        keras.layers.RandomShear(0.2, 0.2),
        keras.layers.RandomZoom(0.2),
        keras.layers.RandomFlip(),
    ]
)

# training_data = training_data.map(lambda x, y: (rescaling_layer(x), y))  # pyright: ignore[reportAttributeAccessIssue]
training_data = training_data.batch(16).map(lambda x, y: (data_aug(x), y))  # pyright: ignore[reportAttributeAccessIssue]


# %%

test_data_dir = pathlib.Path(
    "./dataset/Part 2 - Convolutional Neural Networks (CNN)/dataset/test_set"
)
test_data = keras.utils.image_dataset_from_directory(
    test_data_dir, image_size=(img_h, img_w), batch_size=batch_size
)
print(test_data.class_names)  # pyright: ignore[reportAttributeAccessIssue]

# %%

cnn = keras.models.Sequential(
    [
        keras.layers.Rescaling(1.0 / 255),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPool2D(strides=2),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPool2D(strides=2),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPool2D(strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(num_classes if num_classes > 2 else 1, activation="sigmoid"),
    ]
)

# %%


class AccCallBack(keras.callbacks.Callback):
    def __init__(self, point):
        super(AccCallBack, self).__init__()
        self.point = point

    def on_epoch_end(self, epoch, logs=None):
        acc = logs["accuracy"]  # pyright: ignore[reportOptionalSubscript]
        if acc >= self.point:
            self.model.stop_training = True  # pyright: ignore[reportOptionalMemberAccess]

        return super().on_epoch_end(epoch, logs)


cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
os.makedirs("models", exist_ok=True)
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="models/model.{epoch:02d}-{val_loss:.2f}.keras"),
    AccCallBack(0.98),
]


# %%

cnn.fit(training_data, validation_data=test_data, callbacks=callbacks, epochs=50)

# %%


for i in range(1,5):
    file_path = f"./dataset/Part 2 - Convolutional Neural Networks (CNN)/dataset/single_prediction/cat_or_dog_{i}.jpg"
    test_img = keras.preprocessing.image.load_img(
        file_path,
        target_size=(img_h, img_w),
    )
    test_img = keras.preprocessing.image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)

    pred = cnn.predict(test_img)
    pred_index = 1 if pred[0][0] >= 0.5 else 0
    result_name = class_names[pred_index]
    img = Image.open(file_path)
    img.show()
    print(f"{result_name} - {pred[0][0]}")
    time.sleep(2)
    img.close()
