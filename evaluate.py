from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer

from utils import get_my_cfg
from visdrone import register_one_set

import os


def evaluate(dataset):
    register_one_set(dataset)

    cfg = get_my_cfg()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    model = DefaultTrainer.build_model(cfg)  # just built the model without weights
    checkpoiner = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
    checkpoiner.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)  # loaded the weights we had trained

    evaluator = COCOEvaluator(dataset, ("bbox", ), False, output_dir=os.path.join("output", "evaluate"))
    loader = build_detection_test_loader(cfg, dataset)
    print(inference_on_dataset(model, loader, evaluator))


if __name__ == '__main__':
    evaluate("VisDrone2019-DET-val")