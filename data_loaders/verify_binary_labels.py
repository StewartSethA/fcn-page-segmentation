import numpy as np
import cv2
import sys, os

def verify_binary_labels(folder):
    all_are_binary = True
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if len(filename) > 3 and filename[-4:] == ".png":
            im = cv2.imread(filepath)
            uniq = np.unique(im)
            if len(uniq) > 2 or (0 not in uniq and 255 not in uniq):
                print("File does not have binary labels! Expected only labels 0, 255:", filepath)
                print("Found labels:", np.unique(uniq))
                all_are_binary = False
    return all_are_binary

if __name__ == "__main__":
    verify_binary_labels(sys.argv[1])

