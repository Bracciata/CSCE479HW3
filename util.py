import gdown
from zipfile import ZipFile
import os
import tensorflow as tf            # to specify and run computation graphs

# From https://keras.io/examples/generative/dcgan_overriding_train_step/#prepare-celeba-data
def download_celebs():
    os.makedirs("celeba_gan")

    url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
    output = "celeba_gan/data.zip"
    gdown.download(url, output, quiet=True)
    with ZipFile("celeba_gan/data.zip", "r") as zipobj:
        zipobj.extractall("celeba_gan")

def get_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=32
)
    return dataset.map(lambda x: x / 255.0)
