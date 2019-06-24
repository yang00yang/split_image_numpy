import cv2,numpy as np

def add_noise(img):
    for i in range(20): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img

def add_erode(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    img = cv2.erode(img,kernel)
    return img

def add_dilate(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    img = cv2.dilate(img,kernel)
    return img

if __name__=="__main__":
    img = cv2.imread("../data/test/1.jpg")
    add_dilate()