import easyocr
import numpy as np
import cv2
import cv2.dnn
import random
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import os
import imutils

from ultralytics import YOLO

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
model = YOLO("models/yolo/best.pt")

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

def text_detection(onnx_model, input_image):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape

    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box,
            'scale': scale}
        detections.append(detection)
        draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                            round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

    # Display the image with bounding boxes
    cv2.imwrite('result.jpg', original_image)

    return detections

def crop_and_save_bbox(image_path, bbox, scale):
    # 이미지 로드
    img = cv2.imread(image_path)

    # bbox 좌표 추출
    x, y, w, h = bbox
    
    x1 = int(x * scale)
    y1 = int(y * scale)
    x2 = int((x + w) * scale)
    y2 = int((y + h) * scale)

    # 이미지에서 bbox 영역 crop
    cropped_img = img[y1:y2, x1:x2]
    
    return x1, y1, x2, y2, cropped_img

def get_files_in_directory(directory):
    file_list = []
    for file in os.listdir(directory):
        file_list.append(os.path.join(directory, file))
    return file_list

def mrz_processing(img):
    
    h_org, w_org, _ = img.shape
    
    image = imutils.resize(img, height=600)
    h_resize, w_resize, _ = image.shape
    
    scale = h_org / h_resize
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # smooth the image using a 3x3 Gaussian, then apply the blackhat
    # morphological operator to find dark regions on a light background
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    
    thresh = cv2.threshold(gradX, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # thresh = cv2.adaptiveThreshold(gradX, maxValue=255.0, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV,  blockSize=19,
#  C=9)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=1)
    
    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # loop over the contours
    bbox_infos = []
    for c in cnts:
        # compute the bounding box of the contour and use the contour to
        # compute the aspect ratio and coverage ratio of the bounding box
        # width to the width of the image
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(gray.shape[1])
        # check to see if the aspect ratio and coverage width are within
        # acceptable criteria
        if ar > 5 and crWidth > 0.75:
            # pad the bounding box since we applied erosions and now need
            # to re-grow it
            pX = int((x + w) * 0.02)
            pY = int((y + h) * 0.02)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))
            # extract the ROI from the image and draw a bounding box
            # surrounding the MRZ
            # roi = image[y:y + h, x:x + w].copy()
            bbox_infos.append([int(x * scale),int(y*scale),int(w*scale),int(h*scale)])
    return bbox_infos
    

def main():
    #  yolo detection
    CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    model = YOLO("models/yolo/best.pt")

    test_img_path = 'testimg'
    
    file_list = get_files_in_directory(test_img_path)
    
    output_path = "results"
    for img_path in file_list:
        bboxes = text_detection("models/yolo/best.onnx", img_path)
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        bbox_lists = []
        idx=0
        for info in bboxes:
            idx += 1
            # mrz 부분 제외
            # if info['box'][2]*info['scale'] > width * 0.7:
            #     continue
            crop_result = crop_and_save_bbox(img_path, info['box'], info['scale'])
            bbox_lists.append(crop_result)
        # mrz 부분 전처리
        
        # mrz_info = mrz_processing(img)
        # for mrz in mrz_info:
        #     crop_result = crop_and_save_bbox(img_path, mrz, 1)
        #     bbox_lists.append(crop_result)

        reader = easyocr.Reader(['ko','en'])

        text_results = []
        for bbox in bbox_lists:
            w = bbox[2] - bbox[0]
            result_text = reader.recognize(bbox[4])
            
            text_result = [bbox[0],bbox[1],bbox[2],bbox[3], result_text[0][1]]    
            text_results.append(text_result)

        pil_img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(pil_img)

        fontpath = "ChosunGu.TTF"

        for text_info in text_results:
            x1 = text_info[0]
            y1 = text_info[1]
            x2 = text_info[2]
            y2 = text_info[3]
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            
            text = text_info[4]
            font = ImageFont.truetype(fontpath, 50)
            
            draw.text((x1, y1), text, font=font, fill='black')

        pil_img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        img = cv2.resize(img, dsize=(960,600))
        img2 = cv2.resize(pil_img_cv, dsize=(960,600))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img2)

        width1, height1 = img.size
        width2, height2 = img2.size

        if height1 != height2:
            raise ValueError("이미지의 높이가 일치하지 않습니다.")

        # 두 이미지를 가로로 이어 붙이기
        merged_image = Image.new("RGB", (width1 + width2, height1))
        merged_image.paste(img, (0, 0))
        merged_image.paste(img2, (width1, 0))
        
        name = os.path.split(img_path)[1]
        name = "result_" + name

        merged_image.save(os.path.join(output_path, name))
        print(f"{name} 이 저장되었습니다.")
            
if __name__ == "__main__":
    main()
