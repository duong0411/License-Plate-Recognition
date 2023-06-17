import imutils
from ultralytics import YOLO
import cv2
import numpy as np
import random
from skimage import measure
import matplotlib.pyplot as plt
from imutils import perspective
from skimage.filters import threshold_local
from data_utils import order_points, convert2Square, draw_labels_and_boxes
import pytesseract
#Đoc ảnh
img_path = "duong.jpg"
Ivehicle = cv2.imread(img_path)

#load model
model = YOLO("best.pt","v8")
results = model.predict(show=True, source=[Ivehicle], conf=0.45, save=False)
DP = results[0].numpy()
print(DP)

# opening the file in read mode
my_file = open("coco.txt", "r")

# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

char_list = '0123456789ABCDEFGHKLMNPRSTUVXYZ'
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString
if len(DP) != 0:
    for i in range(len(results[0])):
        print(i)
        boxes = results[0].boxes
        box = boxes[i]  # returns one box
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]
        x = int(bb[0])
        y = int(bb[1])
        w = int(bb[2])
        h = int(bb[3])
        cv2.rectangle(Ivehicle,(int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])),detection_colors[int(clsID)],3,)
        print(x,y,w,h)
        crop_image = Ivehicle[y:h,x:w]
        cv2.imwrite("duong2.jpg",crop_image)


def detect_license_plate(image_path):
            # Đọc ảnh vào dưới dạng grayscale
            image = cv2.imread(image_path, 0)
            preprocessed_image = preprocess_image(image)
            # Sử dụng Tessaract OCR để nhận diện ký tự
            text = pytesseract.image_to_string(preprocessed_image, config='--psm 7')
            # Phân tích kích thước của vùng chứa ký tự
            contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = [cv2.boundingRect(c) for c in contours]
            min_y = min(bounding_boxes, key=lambda box: box[1])[1]
            max_y = max(bounding_boxes, key=lambda box: box[1] + box[3])[1] + \
                    max(bounding_boxes, key=lambda box: box[1] + box[3])[3]

            # Xác định số dòng dựa trên kích thước của vùng chứa ký tự
            if (max_y - min_y) > (2 * (max_y - min_y) / 3):
                num_lines = 2
            else:
                num_lines = 1

            return text, num_lines

def preprocess_image(image):
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        image = cv2.medianBlur(image, 3)
        return image

image_path = "duong2.jpg"

result, num_lines = detect_license_plate(image_path)

print("Biển số:", result)
print("Số dòng:", num_lines)


cv2.imshow("ObjectDetection", Ivehicle)
cv2.imshow("ObjectDetection", crop_image)
cv2.waitKey()
