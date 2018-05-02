import sys,os
import matplotlib.pyplot as plt
from collections import defaultdict

def select_best_thresholds(validationtxts):
    thresholds = []
    fscores = []
    overlap_fscores = []
    class_fscores = defaultdict(list)
    class_accuracies = defaultdict(list)

    for valtxt in sorted(validationtxts):
        with open(valtxt, 'r') as f:
            lines = f.readlines()
            thresh = valtxt[valtxt.index("."):]
            thresh = thresh[:thresh[1:].index(".")+1]
            thresh = float(thresh)
            fscore = float(lines[-2].split(" ")[-1])
            print thresh, fscore
            thresholds.append(thresh)
            fscores.append(fscore)
            overlap_fscores.append(lines[-3].split(" ")[-1])
            print lines[-27-6]
            for l in range(28,27+6):
                line = lines[-l].split(" ")
                print line
                class_fscores[line[0]].append(float(line[-1]))
            print lines[-27-6-6]
            for l in range(22+6+6,27+6+6):
                line = lines[-l].split(" ")
                class_accuracies[line[0]].append(float(line[-1]))
    plt.scatter(thresholds, fscores, label="fscores")
    plt.scatter(thresholds, overlap_fscores, label="overlap_fscores")
    for c in class_fscores.keys():
        plt.scatter(thresholds, class_fscores[c], s=3, label="class"+c+"_fscores")
    #for c in class_accuracies.keys():
    #    plt.scatter(thresholds, class_accuracies[c], label="class"+c+"_accuracies")
    plt.legend()
    plt.show()
    return thresholds, fscores, class_fscores

if __name__ == "__main__":
    vals = os.listdir(sys.argv[1])
    vals = [os.path.join(sys.argv[1], v) for v in vals if "validation" in v and ".txt" in v and v != "validation.txt"]
    print vals
    thresholds, fscores, class_fscores = select_best_thresholds(vals)
    import numpy as np
    best_class_fscores = []
    for c in sorted(class_fscores.keys()):
        fsc = class_fscores[c]
        i = np.argmax(fsc)
        best_threshold = thresholds[i]
        score = fsc[i]
        print "Best threshold for class"+c+":", best_threshold, "with Average F-score:", score
        best_class_fscores.append(score)
    i = np.argmax(fscores)
    best_threshold = thresholds[i]
    score = fscores[i]
    print "Best overall threshold:", best_threshold, "with Average F-score:", score
    mcsf = np.mean(best_class_fscores)
    default_i = thresholds.index(0.5)
    default_score = fscores[default_i]
    print "Improvement by selecting overall threshold instead of default (0.5):", score, (score-default_score)
    print "Improvement by selecting class-specific thresholds:", mcsf, (mcsf-score)
