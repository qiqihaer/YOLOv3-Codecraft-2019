# YOLOv3-Codecraft-2019

本项目为[Codecraft2019决赛](https://github.com/qiqihaer/CodeCraft-2019 "悬停显示")中车牌识别的YOLOv3的训练代码。原始图片来自于[Codecraft2019](https://codecraft.huawei.com "悬停显示")决赛的训练赛。

### trianing-validation data

```plain
└── data
       ├── data   <-- image data
       |   ├── 000000000.png
       |   ├── 000000001.png
       |   └── ...
       |
       └── label.txt  <--- label data: charaters in each image
       |   
       |__ labelGT4000.csv  <--- boxes data: 9 boxes in each image

```

./data/data 和 ./data/label.txt 为Codecraft2019决赛训练赛图片和标签，由华为提供。<br>
./data/labelGT4000.csv 为手标数据

### model 和 pre-trained model

感谢[@ultralystics](https://github.com/ultralytics/yolov3)!!!<br>
model 使用YOLOv3的ssp版本。 pre-trained model 储存在 ./weights 中。
模型和预训练的权重均为[@ultralystics](https://github.com/ultralytics/yolov3)提供。


### 使用
运行train.py

