from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import requests

api_key="***REMOVED***"
url="https://vision.googleapis.com/v1/images:annotate?key="+api_key

request_data = {
  "requests": [
    {
      "image": {
        "content": "/9j/7QBEUGhvdG9zaG9...base64-encoded-image-content...fXNWzvDEeYxxxzj/Coa6Bax//Z"
      },
      "features": [
        {
          "type": "DOCUMENT_TEXT_DETECTION"
        }
      ]
    }
  ]
}

import base64
import json
def send_request(image_path):
    global request_data
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        request_data["requests"][0]["image"]["content"] = encoded_image
        import cv2
        img = cv2.imread(image_path)
        if img.shape[0]*img.shape[1] < 300*1000: # Switch to cheaper text detection for smaller image patches
            print("Doing cheap text detection...")
            request_data["requests"][0]["features"][0]["type"] = "TEXT_DETECTION"
        serialized_request_data = json.dumps(request_data)
        r = requests.post(url, data=serialized_request_data)
        print(r.status_code, r.reason)
        #print(r.text)
        return r.text

def convert_google_api_json_to_bboxes_and_transcriptions(js, confidence_threshold=0.2):
    word_bboxes = []
    word_texts = []
    char_bboxes = []
    char_texts = []
    pages = js["responses"][0]["fullTextAnnotation"]["pages"]
    for page in pages:
        for block in page["blocks"]:
            for paragraph in block["paragraphs"]:
                for word in paragraph["words"]:
                    confidence = word.get("confidence", 0.0)
                    box = word["boundingBox"]["vertices"]
                    b = [int(box[0]["x"]), int(box[0]["y"]), int(box[2]["x"]), int(box[2]["y"])]
                    if b[0] > b[2]:
                        b[0], b[2] = b[2], b[0]
                    if b[1] > b[3]:
                        b[1], b[3] = b[3], b[1]
                    c = float(confidence)
                    if c < confidence_threshold:
                        continue
                    word_bboxes.append(b)
                    word_text = []
                    for symbol in word["symbols"]:
                        confidence = symbol.get("confidence", 0.0)
                        box = symbol["boundingBox"]["vertices"]
                        b = [int(box[0]["x"]), int(box[0]["y"]), int(box[2]["x"]), int(box[2]["y"])]
                        c = float(confidence)
                        #if c < confidence_threshold:
                        #    continue
                        char = symbol["text"]
                        char_bboxes.append(box)
                        char_texts.append(char)
                        word_text.append(char)
                    word_texts.append("".join(word_text))
    return word_bboxes, word_texts, char_bboxes, char_texts

def overlay_simple_google_bboxes(img, result_json):
    textAnnotationsList = result_json["responses"][0]["textAnnotations"]
    for annotation_num, annotation in enumerate(textAnnotationsList):
        box = annotation["boundingPoly"]["vertices"]
        b = [int(box[0]["x"]), int(box[0]["y"]), int(box[2]["x"]), int(box[2]["y"])]
        c = float(annotation_num) / len(textAnnotationsList)
        cs = (int(0),int(255.0*c),int(255.0-255.0*c))
        img = cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), cs, int(2))
    return img

def overlay_google_word_and_char_bboxes(img, result_json):
    # bb = js["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][0]["paragraphs"][0]["words"][0]["symbols"][3]["boundingBox"]
    # tx = js["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][0]["paragraphs"][0]["words"][0]["symbols"][3]["text"]
    # cf = js["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][0]["paragraphs"][0]["words"][0]["confidence"]
    pages = result_json["responses"][0]["fullTextAnnotation"]["pages"]
    for page in pages:
        for block in page["blocks"]:
            for paragraph in block["paragraphs"]:
                for word in paragraph["words"]:
                    confidence = word.get("confidence", 0.0)
                    box = word["boundingBox"]["vertices"]
                    b = [int(box[0]["x"]), int(box[0]["y"]), int(box[2]["x"]), int(box[2]["y"])]
                    c = float(confidence)
                    img = cv2.rectangle(img, (b[0],b[1]),(b[2],b[3]),(0,int(255.0*c),int(255.0-255.0*c)),2)
                    for symbol in word["symbols"]:
                        confidence = symbol.get("confidence", 0.0)
                        box = symbol["boundingBox"]["vertices"]
                        b = [int(box[0]["x"]), int(box[0]["y"]), int(box[2]["x"]), int(box[2]["y"])]
                        c = float(confidence)
                        img = cv2.rectangle(img, (b[0],b[1]),(b[2],b[3]),(0,int(255.0*c),int(255.0-255.0*c)),1)
                        # putText here too!!!
                        cv2.putText(img, symbol["text"], (int((0.75*b[0]+0.25*b[2])), b[3]), cv2.LINE_AA, 1.2, (255, 0, 255), 3)
    return img

if __name__ == "__main__":
    import os
    import cv2
    import numpy as np
    print("Usage: python OCR_google_vision_api.py image [result_json]")
    impath = sys.argv[1]
    if sys.argv[1][-4:] == "json":
        print("Loading existing JSON file:", sys.argv[1])
        with open(sys.argv[1], 'r') as f:
            result_json = json.load(f)
        if len(sys.argv) > 2:
            impath = sys.argv[2]
        else:
            impath = sys.argv[1][:sys.argv[1].find(".")+4]
            print("Loading image from", impath)
    else:
        result = send_request(sys.argv[1])
        with open(sys.argv[1]+"_GoogleVisionOCR.json", 'w') as f:
            f.write(result)
        result_json = json.loads(result)
    text = result_json["responses"][0]["fullTextAnnotation"]["text"]
    with open(impath+"_GoogleVisionOCR.txt", 'w') as f:
        f.write(text)
    img = cv2.imread(impath)
    img = overlay_simple_google_bboxes(img, result_json)
    cv2.imwrite(impath+'_GoogleVisionOCR_output_boxes.jpg', img)

    img = cv2.imread(impath)
    img = overlay_google_word_and_char_bboxes(img, result_json)
    cv2.imwrite(impath+'_GoogleVisionOCR_output_charboxes_wconf.jpg', img)

