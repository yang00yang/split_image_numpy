
# 将图片通过给定的4个坐标进行切割
## 元数据：
```
  {
	"word": "80.35",
	"pos": [{
		"x": 1033,
		"y": 1130
	}, {
		"x": 1089,
		"y": 1130
	}, {
		"x": 1089,
		"y": 1146
	}, {
		"x": 1033,
		"y": 1146
	}]
}, {
	"word": "哈哈",
	"pos": [{
		"x": 1133,
		"y": 1132
	}, {
		"x": 1177,
		"y": 1132
	}, {
		"x": 1177,
		"y": 1151
	}, {
		"x": 1133,
		"y": 1151
	}]
}
```
元数据为这样的json结构，四个点的坐标和对应的字，我们现在要通过这四个坐标进行图片切割

## 核心函数:
```
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
```

### 如何运行
运行bin中的sh文件，参数是data中的子文件夹名，如要切割的数据是 /split/data/20190505。那么执行命令为
```
sh start.sh 20190505
```

### 执行结果
一张图片的执行结果为 一个txt文件和多个图片文件，txt记录了哪个汉字对应的图片名是什么
