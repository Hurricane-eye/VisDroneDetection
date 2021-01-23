from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from visdrone import register_one_set
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from visdrone import get_visdrone_dicts


import os
import cv2

if __name__ == '__main__':
    val = "VisDrone2019-DET-val"
    register_one_set(val)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKER = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = 15000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    predictor= DefaultPredictor(cfg)
    val_dicts = get_visdrone_dicts(val)
    for d in val_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get(val),
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        file_name = d["file_name"].split('/')[3]
        cv2.imwrite(os.path.join("data", val, "images_with_boxes2", file_name), out.get_image()[:,:, ::-1])
        print("Successfully wrote: " + file_name)

    evaluator = COCOEvaluator(val, ("bbox", ), False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, val)
    print(inference_on_dataset(DefaultTrainer.build_model(cfg), val_loader, evaluator))