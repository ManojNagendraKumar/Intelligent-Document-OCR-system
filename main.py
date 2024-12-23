import json

import numpy as np
import cv2 as cv
from pytesseract import pytesseract
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from pdf2image import convert_from_path

script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir,'models','yolov10x_best.pt')
pdf_file = os.path.join(script_dir,'images','cv1.pdf')

model = YOLO(model_path)
dict = None

def process_pdf_to_img(doc_pdf):
    images = convert_from_path(doc_pdf,dpi=300)
    images_path = []
    for i,page in enumerate(images):
        image_path = os.path.join(script_dir,'outputs','img'+str(i+1)+'.jpg')
        page.save(image_path,'JPEG')
        images_path.append(image_path)
    return images_path

def yolo_model(image):
    result = model(image)
    return result

def ocr_processing(image,label,page_num,dict):
    # Pre-process image
    gry = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    bin_img = cv.adaptiveThreshold(gry, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    text = pytesseract.image_to_string(bin_img)
    text = text.replace('\n','')
    text = text.replace('\t','')
    text = text.replace('|','')
    text = text.replace('"','')
    text = text.replace("'", "")
    if dict is None:
        dict = {"Content":[]}
    page_entry = next(
        (entry for entry in dict["Content"] if entry["Page Number"] == page_num),
        None
    )
    if page_entry is None:
        # Add new page entry
        dict["Content"].append({
            "Page Number": page_num,
            "Data": [{"label": label, "text": text}]
        })
    else:
        page_entry["Data"].append({"label": label, "text": text})
    return dict,bin_img

def bounding_boxes(images_path,dict):
    for index in range(len(images_path)):
        image = cv.imread(images_path[index])
        results = yolo_model(image)
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]
                cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                roi = image[y1:y2, x1:x2]
                cv.putText(image, label, (x1, y1 - 5), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
                dict, img = ocr_processing(roi, label,index+1,dict)

    return dict


images_path = process_pdf_to_img(pdf_file)
dict = bounding_boxes(images_path,dict)
document_data = json.dumps(dict, ensure_ascii=False, indent=4)
# image = cv.resize(image,(800,700))
# cv.imshow('window',image)
cv.waitKey(0)