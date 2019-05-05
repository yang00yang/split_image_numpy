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
# 遍历指定目录，显示目录下的所有文件名
count = 0
logger = logging.getLogger("split image")

def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])

def eachFile(rootPath):
    global count
    pathDir = os.listdir(rootPath)
    for allDir in pathDir:
        filePath = os.path.join('%s%s' % (rootPath, allDir))
        if os.path.isdir(filePath):
            # 文件夹
            logger.debug("当前文件夹" + filePath)
            eachFile(filePath + "/")
        else:
            #文件名
            filename = os.path.basename(filePath)
            #后缀
            file = os.path.splitext(filePath)
            aa,type=file
            if (filename.find("2019")>=0) & (type == '.json'):
                count = count +1
                readFile(filePath,rootPath)
            else:
                continue
        #if count>=30:
        #    break
        logger.debug("共有" + str(count) + "个文件")

def moveFileto(sourceDir,  targetDir):
    shutil.copy(sourceDir,  targetDir)

# 读取文件内容并打印
def readFile(filePath,rootPath):
    file = open(filePath, 'r')  # r 代表read
    newfilepath = rootPath[:-10] + "_result/"
    #源图片地址
    imagePath = rootPath + os.path.basename(filePath)[:-4] + "jpg"
    #图片
    image = cv2.imread(imagePath)
    for eachLine in file:
        if image is None:
            continue
        logger.debug("读取到得内容如下：", eachLine)
        result = parseJson(eachLine)
        if result:
            polygens = result['pos']  #坐标
            word = result['word']  #文字
            times = 1
            for image in crop_small_images(image,polygens):
               filename = newfilepath + os.path.basename(filePath)[:-5] + "-" + str(times) + ".jpg"
               cv2.imwrite(filename ,image)
               content = os.path.basename(filePath)[:-5] + "-" + str(times) + ".jpg  " +  word[times-1] + '\n'
               writeFile(filePath,newfilepath, content)
               times += 1
    file.close()


# 切割图片 输入是[[x1,y1,x2,y2,x3,y3,x4,y4],]
def crop_small_images(img,polygens):
    logger.debug("图像：%r" , img.shape)

    cropped_images = []
    for pts in polygens:
        # crop_img = img[y:y+h, x:x+w]
        logger.debug("子图坐标：%r",pts)
        pts_np = np.array(pts)
        pts_np = pts_np.reshape(4,2)
        # print(pts_np)
        min_xy = np.min(pts_np,axis=0)
        max_xy = np.max(pts_np,axis=0)

        # print(min_xy[0],min_xy[1],max_xy[0],max_xy[1])
        crop_img = img[min_xy[1]:max_xy[1],min_xy[0]:max_xy[0]]
        cropped_images.append(crop_img)
    return cropped_images


# 写入备注字段
def writeFile(filePath,newfilepath,word):
    if not os.path.exists(newfilepath):
        os.mkdir(newfilepath)
    # 保存的文件名
    filename = newfilepath + os.path.basename(filePath)[:-4] + "txt"
    fopen = open(filename, 'a+')
    fopen.write('%s%s' % (word, os.linesep))
    fopen.close()

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
    param = ""
    orign = ""
    if len(sys.argv)>=1:
        param = sys.argv[1]
        orign = sys.argv[2]
    # orign = "/Users/admin/Downloads/ocr图片样本/wenzhang/"
    # orign = "/app.fast/projects/split/data"
    rootPath = orign + param + "/"
    eachFile(rootPath)


