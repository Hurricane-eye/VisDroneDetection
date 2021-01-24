from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

from utils import get_my_cfg
from visdrone import register_one_set, get_visdrone_dicts

import cv2
import os


def predict(dataset):
    register_one_set(dataset)

    cfg = get_my_cfg()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    predictor = DefaultPredictor(cfg)
    val_dicts = get_visdrone_dicts(dataset)
    for d in val_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get(dataset),
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        file_name = d["file_name"].split('/')[3]
        cv2.imwrite(os.path.join("data", dataset, "images_with_boxes2", file_name), out.get_image()[:,:, ::-1])
        print("Successfully wrote: " + file_name)


if __name__ == '__main__':
    predict("VisDrone2019-DET-val")