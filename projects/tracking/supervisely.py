import detectron2
from detectron2.utils.logger import setup_logger

from detectron2.structures import BoxMode

import json
import os

setup_logger() # ?

def get_supervisely_dicts(base_dir, sub_dir, categories):

  ann_dir = os.path.join(base_dir, sub_dir, 'ann')
  img_dir = os.path.join(base_dir, sub_dir, 'img')

  anns = sorted(os.listdir(ann_dir))

  dataset_dicts = []
  for idx, ann in enumerate(anns):

    with open(os.path.join(ann_dir, ann)) as f:
      ann_dict = json.load(f)

    record = {}
    
    filename = os.path.join(img_dir, ann[:-5])
    
    record["file_name"] = filename
    record["image_id"] = idx
    record["height"] = ann_dict["size"]["height"]
    record["width"] = ann_dict["size"]["width"]

    objs = []
    for anno in ann_dict["objects"]:
      category = anno["classTitle"]
      if category in categories:
        bbox = anno["points"]["exterior"]
        bbox = bbox[0]+bbox[1]
        bbox = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])]
        obj = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": categories.index(category),
            "iscrowd": 0
        }
        objs.append(obj)
    record["annotations"] = objs
    dataset_dicts.append(record)
  return dataset_dicts

from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.data import DatasetCatalog, MetadataCatalog

def supervisely2coco(dataset_dicts, output_file, dataset_name=None, thing_classes=None):
  
  # it would be better to have detrectron2 functions that accept dataset_dicts not dataset
  
  if dataset_name is None:
    dataset_name = "temp_dataset" # todo: randomize name
  
  DatasetCatalog.register(dataset_name, lambda d=d: dataset_dicts)
  if thing_classes: # todo: we can add it to dataset_dicts, e.g. dataset_dicts[0]['annotations']['class_name'] = ... and here collect
    MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)
    
  convert_to_coco_json("supervisely_val", output_file, allow_cached=False)
    
    
