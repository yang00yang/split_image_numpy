'''
1. 遍历文件夹读取文件夹中的json文件和image图片
2. 通过json信息对image图片进行切割，得到新的图片集
3. 将图片集存入到文件夹中去
'''
import os
import json
import shutil
import cv2
import logging
import time
import sys
import numpy as np
import tqdm
# 遍历指定目录，显示目录下的所有文件名
count = 0
logger = logging.getLogger("split image")

def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])

# 处理src_dir目录中的所有的json文件，对他进行切割
def process_folder(src_dir,dest_dir,label_file_name):
    global count
    file_names = os.listdir(src_dir)
    label_file = open(label_file_name,"a+")
    pbar = tqdm(total=len(file_names))
    i = 0
    for one_file_name in file_names:

        prefix_name = os.path.splitext(one_file_name)#前缀
        subfix_name = os.path.splitext(one_file_name)#后缀

        # 如果不是json文件，直接忽略
        if (subfix_name != '.json'):
            i += 1
            continue #如果不是json文件，直接忽略

        # 看看图片存在不
        image_name = prefix_name + ".jpg"
        image_full_path = os.path.join(src_dir,image_name)
        if not os.path.exists(image_full_path):
            logger.warning("图片路径不存在：%s",image_full_path)
            i+=1
            continue

        process_one_file(src_dir,prefix_name,dest_dir,label_file)
        i += 1
        pbar(i)
    label_file.close()

# 读取文件内容并打印
def process_one_file(src_dir,prefix,dest_dir,label_file):

    json_full_path  = os.path.join(src_dir,prefix+".json")
    image_full_path = os.path.join(src_dir,prefix+".jpg")
    file = open(json_full_path, 'r')  # r 代表read
    image = cv2.imread(image_full_path)

    for line in file:
        if image is None:
            continue
        logger.debug("读取到得内容如下：", line)
        result = parseJson(line)
        if result:
            polygens = result['pos']  #坐标
            words = result['word']  #文字
            times = 1
            for image in crop_small_images(image,polygens):
               filename = os.path.join(dest_dir, prefix + "-" + str(times) + ".jpg")
               cv2.imwrite(filename ,image)
               content = filename + " " + words[times-1] + "\n"
               label_file.write(content)
               times += 1
    file.close()


# 切割图片 输入是[[x1,y1,x2,y2,x3,y3,x4,y4],]
def crop_small_images(img,polygens):
    logger.debug("图像：%r" , img.shape)

    cropped_images = []
    for pts in polygens:
        # crop_img = img[y:y+h, x:x+w]
        # logger.debug("子图坐标：%r",pts)
        pts_np = np.array(pts)
        pts_np = pts_np.reshape(4,2)
        # print(pts_np)
        min_xy = np.min(pts_np,axis=0)
        max_xy = np.max(pts_np,axis=0)

        # print(min_xy[0],min_xy[1],max_xy[0],max_xy[1])
        crop_img = img[min_xy[1]:max_xy[1],min_xy[0]:max_xy[0]]
        cropped_images.append(crop_img)
    return cropped_images


# 解析json结构获得坐标
def parseJson(eachLine):
    jsonLine = json.loads(eachLine)
    prism_wordsInfo = jsonLine['prism_wordsInfo']
    posList = []
    wordList = []
    result = {}
    for info in prism_wordsInfo:
        dataList = []
        word = info['word']
        wordList.append(word)
        for pos in info['pos']:
            x = pos['x']
            y = pos['y']
            dataList.append(x)
            dataList.append(y)
        posList.append(dataList)
    result['pos'] = posList
    result['word'] = wordList
    return result


if __name__ == '__main__':
    init_logger()
    src_dir = None
    dst_dir = None
    if len(sys.argv)==3:
        src_dir = sys.argv[1]
        dst_dir = sys.argv[2]
    else:
        logger.debug("参数格式错误：src_dir dst_dir label_name")
        exit(-1)

    label_name = "label.txt"
    if not sys.argv.get(3,None):
        label_name = sys.argv[3]


    if not os.path.exists(src_dir):
        logger.error("源目录%s不存在")
        exit(-1)
    if not os.path.exists(dst_dir):
        logger.error("目标目录%s不存在")
        exit(-1)

    # 处理src_dir目录中的所有的json文件，对他进行切割
    logger.debug("源文件夹:%s,目标文件夹:%s,标签名字：%s",src_dir,dst_dir,label_name)
    process_folder(src_dir,dst_dir,os.path.join(dst_dir,label_name))


