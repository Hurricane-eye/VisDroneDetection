from detectron2.config import get_cfg
from detectron2 import model_zoo

import os

def get_my_cfg():

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATALOADER.NUM_WORKER = 4
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = 15000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "train")

    return cfg