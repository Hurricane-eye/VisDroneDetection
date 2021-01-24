from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer

from utils import get_my_cfg
from visdrone import register_one_set

import os


def evaluate(dataset):
    register_one_set(dataset)

    cfg = get_my_cfg()
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    evaluator = COCOEvaluator(dataset, ("bbox", ), False, output_dir=os.path.join("output", "evaluate"))
    loader = build_detection_test_loader(cfg, dataset)
    print(inference_on_dataset(DefaultTrainer.build_model(cfg), loader, evaluator))


if __name__ == '__main__':
    evaluate("VisDrone2019-DET-val")