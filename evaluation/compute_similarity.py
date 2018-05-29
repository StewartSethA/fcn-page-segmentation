from __future__ import print_function
import os
import sys
import numpy as np
import cv2

def intersect(map1, map2):
    return np.multiply(np.clip(map1, 0, 1), np.clip(map2, 0, 1))
    
def unite(map1, map2):
    return np.clip(map1 + map2, 0, 1)

def IoU(map1, map2):
    intersection = intersect(map1, map2)
    union = unite(map1, map2)
    intersection_area = np.count_nonzero(intersection)
    union_area = np.count_nonzero(union)
    iou = float(intersection_area) / float(union_area) if union_area > 0 else 1
    # We define IoU for both maps being blank as 1; the predictions match so neither should be penalized.
    return iou
    
def get_precision(gt, pred):
    intersection = intersect(gt, pred)
    true_positive = np.count_nonzero(intersection)
    false_positive = np.count_nonzero(pred) - true_positive
    pred_count = np.count_nonzero(pred)
    precision = (float(true_positive) / float(pred_count)) if pred_count > 0 else 1.0
    return precision
    
def get_recall(gt, pred):
    intersection = intersect(gt, pred)
    true_positive = np.count_nonzero(intersection)
    gt_count = np.count_nonzero(gt)
    recall = (float(true_positive) / float(gt_count)) if gt_count > 0 else 1.0
    return recall

def get_scores(gt, pred):
    precision = get_precision(gt, pred)
    recall = get_recall(gt, pred)
    iou = IoU(gt, pred)
    f_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f_score, iou
    
if __name__ == "__main__":
    authorsdir = "Authors"
    if len(sys.argv) > 1:
        authorsdir = sys.argv[1]
    else:
        print("Usage: python compute_similarity.py <authorsdir> <threshold>")
    threshold = 1
    if len(sys.argv) > 2:
        threshold = int(sys.argv[2])
    authors = os.listdir(authorsdir)
    originals = os.listdir("Originals")
    
    from collections import defaultdict
    class2class_fscores = defaultdict(lambda:0.0)
    class2class_counts = defaultdict(lambda:0.0)
    class_overall_ious = defaultdict(lambda:0.0)
    mean_ious = defaultdict(lambda:0.0)
    extreme_ious = defaultdict(lambda:0.0)
    pct_pixels_byclass_allauthors = defaultdict(lambda:0.0)
    # Table: Author_Class to Image_Author
    table = defaultdict(lambda:defaultdict(lambda:0.0))
    
    for original in originals:
        image_path = os.path.join("Originals", original)
        print("")
        print("=========================================")
        print("ORIGINAL:", image_path)
        if os.path.exists(image_path):
            image = cv2.imread(image_path, 0)
        else:
            print("Image does not exist:", image_path)
        filebase = os.path.splitext(os.path.basename(image_path))[0]
        content_type_to_idx = {"dotted_lines":0, "handwriting":1, "machine_print":2, "solid_lines":3, "stamps":4}
        for content_type in ["machine_print", "handwriting", "dotted_lines", "solid_lines", "stamps"]:
            content_originals = []
            print("")
            print("Content type:", content_type)
            intersection_prior = np.ones(image.shape)
            union_prior = np.zeros(image.shape)
            author_channels = []
            for author_num, author in enumerate(authors):
                
                channel_path = os.path.join(authorsdir, author, filebase + "_" + content_type + ".png")
                if not os.path.exists(channel_path):
                    print("Not found:", channel_path)
                    channel_path = os.path.join(authorsdir, author, filebase + "_" + str(content_type_to_idx[content_type]) + ".png")
                    print("Attempting new channel path:", channel_path)
                channel = cv2.imread(channel_path, 0)
                T, channel = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY) # Just in case it's not binary yet...
                # Aggregate into intersection and union.
                author_channels.append(channel)
                channel_pixels = np.count_nonzero(channel)
                pct_pixels_byclass_allauthors[content_type] += channel_pixels
                print("Image",image_path,"Author:",author,"Channel",content_type,"Number of pixels in class:", channel_pixels)
                content_originals.append(channel.astype('float32')/255.0)
                intersection_prior = intersect(intersection_prior, channel)
                union_prior = unite(union_prior, channel)
                
                if author_num != 0:
                    intersection_area = np.count_nonzero(intersection_prior)
                    union_area = np.count_nonzero(union_prior)
                    iou = float(intersection_area) / float(union_area) if union_area > 0 else 1.0
                    print("Original:", original, "Author:", author, "Channel:", content_type, "Cumulative IoU:", iou)
                    
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # Match against * other * authors as though they are GT.
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                print("----------- PERFORMING NESTED COMPARISON... --------------")
                for author_num2, author2 in enumerate(authors):
                
                    channel2_path = os.path.join(authorsdir, author2, filebase + "_" + content_type + ".png")
                    if not os.path.exists(channel2_path):
                        print("Not found:", channel2_path)
                    channel2 = cv2.imread(channel2_path, 0)
                    T, channel2 = cv2.threshold(channel2, 127, 255,cv2.THRESH_BINARY) # Just in case it's not binary yet...
                    
                    scores = get_scores(channel, channel2)
                    print("Author", author_num, "to", author_num2, "Scores:")
                    print("   Precision", scores[0])
                    print("   Recall", scores[1])
                    print( "   F-Score", scores[2])
                    print("   IoU", scores[3])
                    table[str(author_num)+"_"+content_type][image_path+"_"+str(author_num2)] = scores[2]
                    if author_num != author_num2:
                        class2class_fscores[content_type] += scores[2]
                        class2class_counts[content_type] += 1
                        mean_ious[content_type] += scores[3]
                
            
            content_originals = np.array(content_originals)
            print(content_originals.shape)
            content_flat = np.sum(content_originals, axis=0)
            consensus_rgb = np.zeros((content_flat.shape[0], content_flat.shape[1], 3))
            
            wblbp = False
            rgb_blended = True
            if rgb_blended:
                for a in range(len(author_channels)):
                    consensus_rgb[:,:,a] += author_channels[a]

            if wblbp:
                for i in range(3):
                    consensus_rgb[:,:,i] = 1.0 - np.less(0, content_flat).astype('float32')
                consensus_rgb[:,:,0] += np.equal(2, content_flat).astype('float32')
                consensus_rgb[:,:,1] += np.equal(2, content_flat).astype('float32') * 0.75
                #consensus_rgb[:,:,2] += np.equal(2, content_flat).astype('float32') * 0.5
                consensus_rgb[:,:,0] += np.equal(3, content_flat).astype('float32')
                consensus_rgb[:,:,2] += np.equal(3, content_flat).astype('float32')
                #consensus_rgb[:,:,2] = np.equal(2, content_flat).astype('float32')
            cv2.imwrite(original+"_" + content_type + "_colored.png", consensus_rgb * 255)
            
            content_originals = np.mean(content_originals, axis=0)
            #cv2.imshow("Content originals", content_originals)
            cv2.imwrite(original+"_" + content_type + "_averaged.png", content_originals * 255)
            
            consensus_and_conflict = np.zeros((image.shape[0], image.shape[1], 3))
            consensus_and_conflict[:,:,1] += intersection_prior
            consensus_and_conflict[:,:,2] += union_prior - intersection_prior
            
            #cv2.imshow("Consensus and Conflict", consensus_and_conflict)
            cv2.imwrite(original+"_" + content_type + "_consensus-and-conflict.png", consensus_and_conflict * 255)
            cv2.imwrite(original+"_"+content_type+"_union.png", union_prior*255)
            cv2.imwrite(original+"_"+content_type+"_intersection.png", intersection_prior*255)
            #cv2.waitKey(10)
    
    # Summarize the mutual errors by content type.
    for content_type in class2class_fscores.keys():
        avg_fscore = class2class_fscores[content_type] / class2class_counts[content_type]
        print("Class", content_type, "Average F-score:", avg_fscore)
    print("Overall average F-score:", np.mean(list(class2class_fscores.values()))/np.mean(list(class2class_counts.values())), np.std(list(class2class_counts.values())))
    for content_type in class2class_fscores.keys():
        mean_iou = mean_ious[content_type] / class2class_counts[content_type]
        print("Class", content_type, "Mean IoU:", mean_iou)

    for content_type in pct_pixels_byclass_allauthors.keys():
        print("Total Pixels in class", content_type, ":", pct_pixels_byclass_allauthors[content_type])

    # Print the F-score table.
    
    print(table.keys())
    print("")
    sorted_rows = sorted(table.keys())
    i = 0
    for row_idx in sorted_rows:
        if i == 0:
            sorted_cols = sorted(table[row_idx].keys())
            header = "\t" + ("\t".join(sorted_cols))
            print("Header:")
            print(header)
        if False: # TODO: Broken!
               row = [table[row_idx][col] for col in sorted_cols]
               print(row_idx, i)
               print(row)
               line = row_idx + "\t" + ("\t".join(row))
               print("Line:")
               print(line)
        i += 1

