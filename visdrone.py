from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

import cv2
import os
import random


CLASS_NAMES = ['__background__',  # always index 0
               'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']


def get_visdrone_dicts(dataset):
    root = os.path.join("data", dataset)
    images_names = list(sorted(os.listdir(os.path.join(root, "images"))))
    annotations_names = list(sorted(os.listdir(os.path.join(root, "annotations"))))
    dataset_dicts = []
    num_images = len(images_names)
    for idx in range(num_images):
        image_path = os.path.join(root, "images", images_names[idx])
        annotation_path = os.path.join(root, "annotations", annotations_names[idx])
        height, width = cv2.imread(image_path).shape[:2]
        # otice: "file_name" is image's path, not only the name
        record = {"file_name": image_path, "image_id": idx, "height": height, "width": width}
        objs = []
        with open(annotation_path, "r") as file:
            try:
                for annotation in file:
                    annotation = list(map(int, annotation.rstrip("\n").split(',')))
                    obj = {
                        "bbox": [annotation[0], annotation[1], annotation[2], annotation[3]],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": annotation[5]
                    }
                    objs.append(obj)
            finally:
                file.close()
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_one_set(dataset):
    DatasetCatalog.register(dataset, lambda: get_visdrone_dicts(dataset))
    MetadataCatalog.get(dataset).set(thing_classes=CLASS_NAMES)


def verify_dataset(dataset):
    dataset_dicts = get_visdrone_dicts(dataset)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=MetadataCatalog.get(dataset),
                                scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    verify_dataset("VisDrone2019-DET-train")


