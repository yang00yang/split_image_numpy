import cv2,os,numpy as np

blur_method = ["2D","均值","高斯","中值","双边"]

def main():
    for f in os.listdir("data/test"):
        if not (os.path.exists("data/test_result")):
            os.makedirs("data/test_result")
        f_name = os.path.join("data/test",f)
        img = cv2.imread(f_name)

        save_image(f,"原图",img)
        for method in blur_method:
            blur_img = blur(method, img, f)
            sharpen_img = sharpen(blur_img)
            save_image(f,method,sharpen_img)

def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def save_image(original_name,type,dst_image):
    prefix,subfix = os.path.splitext(original_name)
    f_name = os.path.join("data/test_result",prefix+"_"+type+subfix)
    print("保存图像：",f_name)
    cv2.imwrite(f_name,dst_image)


def blur(type,img,name):
    dst_image = None
    if type == "2D" : dst_image = filer2d(img)
    if type == "均值": dst_image = filer_avg(img)
    if type == "高斯": dst_image = filter_gaussian(img)
    if type == "中值": dst_image = filter_median(img)
    if type == "双边": dst_image = filter_bi(img)
    return dst_image


def filer2d(image):
    print("2D滤波器")
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)


def filer_avg(img):
    print("均值滤波器")
    blur = cv2.blur(img, (3, 5))  # 模板大小为3*5, 模板的大小是可以设定的
    box = cv2.boxFilter(img, -1, (3, 5))
    return blur

def filter_gaussian(img):
    print("高斯滤波器")
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # （5,5）表示的是卷积模板的大小，0表示的是沿x与y方向上的标准差
    return blur

def filter_median(img):
    print("中值滤波器")
    blur = cv2.medianBlur(img, 5)  # 中值滤波函数
    return blur

def filter_bi(img):
    print("双边滤波器")
    blur = cv2.bilateralFilter(img, 9, 80, 80)
    return blur

if __name__ == "__main__":
    main()