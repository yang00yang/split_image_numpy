#refer:https://blog.csdn.net/JohinieLi/article/details/80432851
import numpy as np,cv2
import matplotlib.pyplot as plt
from skimage import exposure,data

image =data.camera()*1.0
image = cv2.imread("../data/test/1.jpg")
hist1=np.histogram(image, bins=2)
hist2=exposure.histogram(image, nbins=2)
# print(hist1)
# print(hist2)

# 画出图的直方图
plt.figure("hist")
#print(image.shape)
arr=image.flatten()
#print(arr.shape)
n, bins, patches = plt.hist(arr, bins=256, normed=1,edgecolor='None',facecolor='red')
plt.show()

# 均衡化图片，就是把个通道的值均摊到整个颜色值上
target_image = exposure.equalize_hist(image,nbins=256)
plt.imshow(target_image)
plt.show()

import skimage
# img = skimage.data.immunohistochemistry()
img = image
skimage.io.imshow(img)
skimage.io.show()

img_histeq = skimage.exposure.equalize_adapthist (img,20)
skimage.io.imshow(img_histeq)
skimage.io.show()

img_gamma = skimage.exposure.adjust_gamma(img, gamma=0.5, gain=1)
skimage.io.imshow(img_gamma)
skimage.io.show()

img_sigmoid = skimage.exposure.adjust_sigmoid(img)
skimage.io.imshow(img_sigmoid)
skimage.io.show()



