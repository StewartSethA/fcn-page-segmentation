from __future__ import print_function
import matplotlib.pyplot as plt

def load_image(image_path):
    img = plt.imread(image_path)
    return img
