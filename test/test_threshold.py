import numpy as np
import cv2

image = cv2.imread("../data/test/2.jpg")#读取图像
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#将图像转化为灰度
blurred = cv2.GaussianBlur(image, (5, 5), 0)#高斯滤波
cv2.imshow("Image", image)

#自适应阈值化处理
#cv2.ADAPTIVE_THRESH_MEAN_C：计算邻域均值作为阈值
thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Mean Thresh", thresh)
#cv2.ADAPTIVE_THRESH_GAUSSIAN_C：计算邻域加权平均作为阈值
thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gaussian Thresh", thresh)
cv2.waitKey(0)