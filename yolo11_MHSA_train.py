import warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 必须放在所有导入之前！

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    #     修改为自己的配置文件地址
    model = YOLO('ultralytics/cfg/models/11/yolov11-MHSA.yaml')
    #model.load('yolo11n.pt')
    #     修改为自己的数据集地址
    model.train(data='accident_DDD.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                single_cls=False,  # 是否是单类别检测
                batch=8,
                close_mosaic=10,
                workers=0,
                optimizer='SGD',
                amp=True,
                project='runs/yolo11_MHSA_train',
                name='MHSA',
                )