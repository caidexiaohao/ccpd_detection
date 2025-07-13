# detect_yolo.py
from ultralytics import YOLO
import cv2, os

model = YOLO(r"runs\detect\train\weights\best.pt")
input_folder = "datasets/images/"
crop_folder = "recognition/images/"
os.makedirs(crop_folder, exist_ok=True)
for subset in os.listdir(input_folder):
    subset_input_folder = os.path.join(input_folder, subset)
    subset_crop_folder = os.path.join(crop_folder, subset)
    
    if not os.path.exists(subset_input_folder):
        continue
        
    os.makedirs(subset_crop_folder, exist_ok=True)
    for img_name in os.listdir(subset_input_folder):
        path = os.path.join(subset_input_folder, img_name)
        img = cv2.imread(path)
        result = model.predict(path, conf=0.5)[0]
        for i,box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            out_path = os.path.join(subset_crop_folder, f"{img_name}")
            cv2.imwrite(out_path, crop)

