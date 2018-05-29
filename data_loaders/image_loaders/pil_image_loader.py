from __future__ import print_function
from PIL import Image
import numpy as np

def load_image(image_path):
    img = np.array(Image.open(image_path))
    return img
