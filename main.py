from __future__ import print_function

from util import *
from model import *
import os.path 

if not os.path.isdir('celeba_gan'):
    download_celebs() 

ds = get_dataset()

model = Model()
batch_size = 32
model.train(ds)
