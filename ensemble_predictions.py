import os
import cv2
import numpy as np
from collections import defaultdict

def single_model_ensemble(model_history):
    # Takes a history of snapshots of the same model
    
    # Create a heatmap of per-pixel accuracy over time. Did we ever get it right, or not?
    
    # Solve the rolling ball problem!!!
    pass

def combine_predictions(base_folder, subfolder_name, output_folder='./', mode='median', classes=['0','1','2','3','4']):
    folders = os.listdir(base_folder)
    folders = [os.path.join(base_folder, f, subfolder_name) for f in folders]
    folders = [f for f in folders if os.path.isdir(f)]
    #print(folders)
    predictions = defaultdict(lambda:None)
    predcounts = defaultdict(lambda:0)
    for folder in folders:
        #print(folder)
        ims = os.listdir(folder)
        ims = [os.path.join(folder, im) for im in ims if ".jpg" == im[-4:]]
        for im in ims:
            #print(im)
            for clas in classes:
                gtp = im.replace(".jpg", "_" + clas + ".png")
                print(gtp)
                try:
                    gt = cv2.imread(gtp, 0).astype('float32')/255
                    print("READ", gtp)
                except:
                    continue
                if predictions[os.path.basename(gtp)] is None:
                    predictions[os.path.basename(gtp)] = gt
                    predcounts[os.path.basename(gtp)] = 1
                else:
                    predictions[os.path.basename(gtp)] += gt
                    predcounts[os.path.basename(gtp)] += 1
                
    for pp, pred in predictions.items():
        pred = pred / predcounts[pp]
        cv2.imwrite(os.path.join(output_folder, os.path.basename(pp)), pred*255)
        
if __name__ == "__main__":
    import sys
    combine_predictions(sys.argv[1], sys.argv[2], sys.argv[3])
