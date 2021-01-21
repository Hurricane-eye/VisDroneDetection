from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from visdrone_voc import *
import os
import cv2
import torch

register_visdrone_voc('VISDRONE_VOC', os.path.join('/home/chenzhengxi/data/VisDrone/VisDrone2018-DET-train'),
                      'train', 2012)
register_visdrone_voc('VISDRONE_VAL', os.path.join('/home/chenzhengxi/data/VisDrone/VisDrone2018-DET-val'),
                      'val', 2012)
register_visdrone_voc('VISDRONE_TEST', os.path.join('/home/chenzhengxi/data/VisDrone/VisDrone2019-DET-test-dev'),
                      'test', 2012)
cfg = get_cfg()
cfg.merge_from_file('configs/faster_rcnn_X_101_32x8d_FPN_3x.yaml')

# cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
#resume=True可继续训练并加载最新权重
trainer.resume_or_load(resume=False)
trainer.train()

#以下代码可指定具体权重
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0229999.pth")
#checkpointer = DetectionCheckpointer(trainer.model)
#checkpointer.load(cfg.MODEL.WEIGHTS)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model

evaluator = PascalVOCDetectionEvaluator(cfg.DATASETS.TEST[0])
# val_loader = build_detection_test_loader(cfg, "VISDRONE_VAL")
# result_val = inference_on_dataset(trainer.model, val_loader, evaluator)
# print(result_val)
print(trainer.test(cfg, trainer.model, evaluator))

# predictor = DefaultPredictor(cfg)
# im = cv2.imread('/home/chenzhengxi/data/VisDrone/VisDrone2018-DET-val/JPEGImages/0000026_03500_d_0000031.jpg')
# outputs = predictor(im)
# ooo = outputs['instances'].to(torch.device("cpu"))
# boxes = ooo.pred_boxes.tensor.numpy()
# print(boxes)
# for i in range(len(boxes)):
#     cv2.rectangle(im, tuple(boxes[i, 0:2]), tuple(boxes[i, 2:4]), (0, 255, 0), 2)
#
# cv2.imshow('visdrone', im)
# cv2.waitKey(0)