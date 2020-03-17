import os
import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog

def get_img_dicts(img_dir):

  img_files = sorted(os.listdir(img_dir))

  img0 = cv2.imread(os.path.join(img_dir, img_files[0]))
  height, width = img0.shape[:2]

  dataset_dicts = []
  for idx, img_file in enumerate(img_files):

    record = {}    
    record["file_name"] = os.path.join(img_dir, img_file)
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width

    objs = []
    record["annotations"] = objs
    dataset_dicts.append(record)

  return dataset_dicts

DatasetCatalog.clear()
DatasetCatalog.register("inference_dataset", lambda: get_img_dicts('/content/LhcTGafSgw_all'))
MetadataCatalog.get("inference_dataset").set(thing_classes=["person"])

from detectron2.config import get_cfg

cfg = get_cfg()

# Cascade Mask RCNN ResNeXt 152
cfg.merge_from_file("./detectron2_repo/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
cfg.MODEL.WEIGHTS = "/content/drive/My Drive/respo/MOT/cascade_rcnn/model_0018999.pth" # model_0005999 model_final.pth

cfg.MODEL.MASK_ON = False
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.DATASETS.TEST = ("inference_dataset",)
cfg.DATALOADER.NUM_WORKERS = 4

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

model = build_model(cfg)
model.eval()
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.data import build_detection_test_loader

val_dataset = cfg.DATASETS.TEST[0]
output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

coco_data_loader = build_detection_test_loader(cfg, val_dataset) # batch_size = cfg.DATALOADER.NUM_WORKERS
coco_evaluator = COCOEvaluator(val_dataset, cfg, False, output_folder)

inference_on_dataset(model, coco_data_loader, evaluator=coco_evaluator)

import json

with open('/content/output/inference/inference_dataset_coco_format.json', 'r') as f:
  coco_dict = json.load(f)

with open('/content/output/inference/coco_instances_results.json', 'r') as f:
  annotations_list = json.load(f)

for idx, ann in enumerate(annotations_list):
  ann['id'] = idx

coco_dict['annotations'] = annotations_list

with open('/content/output/inference/inference_coco.json', 'w') as f:
  json.dump(coco_dict, f)
