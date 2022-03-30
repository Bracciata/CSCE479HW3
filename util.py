import gdown
from zipfile import ZipFile
import os
# From https://keras.io/examples/generative/dcgan_overriding_train_step/#prepare-celeba-data
def download_celebs():
    os.makedirs("celeba_gan")

    url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
    output = "celeba_gan/data.zip"
    gdown.download(url, output, quiet=True)
    with ZipFile("celeba_gan/data.zip", "r") as zipobj:
        zipobj.extractall("celeba_gan")