import cv2
from predictorClass import Predictor

predictor = Predictor(weights_path="best_fpn.h5")
img = cv2.imread("img/img.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pred = predictor(img, None)
pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
cv2.imwrite("submit/img.jpg", pred)
