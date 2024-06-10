import pdb

from ultralytics import YOLOWorld
import cv2
import os


# Disable cuda
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load classes in txt file. Each class is a string on a new line
with open(
    "/home/sacha/Documents/cg-plus/model_checkpoints/scannet200_classes.txt", "r"
) as f:
    classes = f.read().splitlines()

# Load test image
img = cv2.imread("/home/sacha/Downloads/Replica/room0/results/frame000000.jpg")

# Flip to RGB
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


model = YOLOWorld("/home/sacha/Documents/cg-plus/model_checkpoints/yolov8s-world.pt")
model.set_classes(classes)
model.to("cpu")

import time

start = time.time()
for _ in range(1000):
    output = model.predict(img)
stop = time.time()
# Print fps
print(1000 / (stop - start))
import pdb

pdb.set_trace()

bbox = output[0].boxes.xyxy.cpu().numpy()

output[0].show()
