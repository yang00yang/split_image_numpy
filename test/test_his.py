#refer:https://blog.csdn.net/JohinieLi/article/details/80432851
import numpy as np,cv2
import matplotlib.pyplot as plt
from skimage import exposure,data
image =data.camera()*1.0
image = cv2.imread("../data/test/1.jpg")
hist1=np.histogram(image, bins=2)
hist2=exposure.histogram(image, nbins=2)
print(hist1)
print(hist2)


plt.figure("hist")
print(image.shape)
arr=image.flatten()
print(arr.shape)
n, bins, patches = plt.hist(arr, bins=256, normed=1,edgecolor='None',facecolor='red')
plt.show()