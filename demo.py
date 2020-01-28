from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections
import mmcv
from mmcv import Config
from mmdet.datasets import get_dataset
import cv2
import os
# RoITransformer
config_file = 'configs/DOTA/faster_rcnn_r50_fpn_1x_dota_RoITrans_v2_ms.py'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_dota_RoITrans_v2_ms/epoch_12.pth'

# mask_rcnn_r50_fpn_1x_dota_cy
cfg = Config.fromfile(config_file)
data_test = cfg.data['test']
dataset = get_dataset(data_test)
classnames = dataset.CLASSES

model = init_detector(config_file, checkpoint_file, device='cuda:0')

imgname = 'demo/test_img/P0006_crop.png'
result = inference_detector(model, imgname)
print('result: ', result)
img = draw_poly_detections(imgname, result, classnames, scale=1, threshold=0.001)
outimg = 'demo/test_img/P0006_crop_test.png'
cv2.imwrite(outimg, img)