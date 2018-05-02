import Image
from pytesseract import pytesseract as pt
#from pytesseract import image_to_string
import os, sys

# https://stackoverflow.com/questions/20831612/getting-the-bounding-box-of-the-recognized-words-using-python-tesseract
import csv
import cv2

#pt.run_tesseract(sys.argv[1], 'output', lang='eng', boxes=True, config='hocr')

#def write_chars_and_bboxes_to_text(img):

# https://pypi.python.org/pypi/pytesseract

if __name__ == "__main__":
    string = pt.image_to_string(Image.open(sys.argv[1]), lang='eng').replace("\n\n", "\n")
    print(string)
    text_file = "masked/form1_transcription.txt"
    text = ""
    with open(text_file, 'r') as f:
        text = "".join(f.readlines())
    
    char_and_box_lines = pt.image_to_boxes(Image.open(sys.argv[1])) #, 'output', lang='eng', boxes=True, config='hocr')

    #print pt.image_to_data(Image.open(sys.argv[1]))

    char_and_box_lines = char_and_box_lines.split("\n")
    #for line in char_and_box_lines:
    #    print line

    char_and_box_lines = [line.split(" ") for line in char_and_box_lines]
    chars = [l[0] for l in char_and_box_lines]
    boxes = [(int(l[1]),int(l[2]),int(l[3]),int(l[4])) for l in char_and_box_lines]

    '''
    boxes = []
    with open('output.box', 'rb') as f:
        reader = csv.reader(f, delimiter = ' ')
        for row in reader:
            if len(row) == 6:
                boxes.append(row)
    '''

    # Draw the bounding boxes
    img = cv2.imread(sys.argv[1])
    h,w,_ = img.shape
    for boxnum, b in enumerate(boxes):
        c = float(boxnum) / len(boxes)
        img = cv2.rectangle(img, (b[0],h-b[1]),(b[2],h-b[3]),(255.0*c,0,255.0-255.0*c),2)
        #img = cv2.rectangle(img, (int(b[1]),h-int(b[2])),(int(b[3]),h-int(b[4])),(255,0,0),2)
    cv2.imwrite(sys.argv[1]+'_output_boxes.jpg', 255*img)
    #cv2.waitKey()
    import json
    result_json = {"boxes":boxes, "chars": chars, "text":text}
    with open(sys.argv[1]+"_TesseractOCR.json", 'w') as f:
        json.dump(result_json, f, indent=2)
    with open(sys.argv[1]+"_TesseractOCR.txt", 'w') as f:
        f.write(text)
