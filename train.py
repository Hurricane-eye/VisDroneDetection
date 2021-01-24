from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo

from utils import get_my_cfg
from visdrone import register_one_set

import os

if __name__ == '__main__':
    train_dataset = "VisDrone2019-DET-train"
    val_dataset = "VisDrone2019-DET-val"
    register_one_set(train_dataset)
    register_one_set(val_dataset)

    cfg = get_my_cfg()
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TRAIN = (train_dataset, )
    cfg.DATASETS.TEST = (val_dataset, )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator(val_dataset, ("bbox",), False, output_dir=os.path.join("output", "evaluate"))
    trainer.test(cfg, trainer.model, evaluator)