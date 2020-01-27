import json
import os

import cv2
import numpy as np

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.structures import BoxMode

from detectron2.data import DatasetCatalog, MetadataCatalog

CROP_SRC_DIR = os.path.join('/content',
                       'input_new_pseudolabel_assorted_pseudolabel_runOne_ekstraklasa_kolejki_201819_set1-imgs_0_5000',
                       'pseudolabel_runOne_ekstraklasa_kolejki_201819_set1-imgs_0_5000',
                       'ekstraklasa_kolejki_201819_set1',
)

SEQUENCES = {
'train': [
{'type': 'crop', 'dir': ['ocr-combined-pseudolabel_runOne_ekstraklasa_kolejki_201819_set1-imgs_0_5000-part_3',
'pseudolabel_runOne_ekstraklasa_kolejki_201819_set1-imgs_0_5000-part_3']},
{'type': 'crop', 'dir': ['ocr-combined-pseudolabel_runOne_ekstraklasa_kolejki_201819_set1-imgs_0_5000-part_1',
'pseudolabel_runOne_ekstraklasa_kolejki_201819_set1-imgs_0_5000-part_1']},
{'type': 'crop', 'dir': ['ocr-combined-pseudolabel_runOne_ekstraklasa_kolejki_201819_set1-imgs_0_5000-part_2',
'pseudolabel_runOne_ekstraklasa_kolejki_201819_set1-imgs_0_5000-part_2']},
{'type': 'full', 'dir': ['full-ocr-combined-pseudolabel_ekstraklasa_kolejki_2018_2019_filtered_set2_vids0_50_idx_0_1000',
'pseudolabel_ekstraklasa_kolejki_2018_2019_filtered_set2_vids0_50_idx_0_1000']},
{'type': 'full', 'dir': ['full-ocr-combined-pseudolabel_ekstraklasa_kolejki_2018_2019_filtered_set2_vids0_50_idx_1000_2000',
'pseudolabel_ekstraklasa_kolejki_2018_2019_filtered_set2_vids0_50_idx_1000_2000']},
{'type': 'full', 'dir': ['full-ocr-combined-pseudolabel_ekstraklasa_kolejki_2018_2019_filtered_set2_vids0_50_idx_2000_3394',
'pseudolabel_ekstraklasa_kolejki_2018_2019_filtered_set2_vids0_50_idx_2000_3394']},
],
'val': [
{'type': 'crop', 'dir': ['ocr-combined-pseudolabel_runOne_ekstraklasa_kolejki_201819_set1-imgs_0_5000-part_0',
'pseudolabel_runOne_ekstraklasa_kolejki_201819_set1-imgs_0_5000-part_0']},
]
}

def get_supervisely_dicts_cropped(sequences, dataset_type):

  IMAGE_DIR = '/content/detections'
  os.makedirs(IMAGE_DIR, exist_ok=True)
  dst_dir = os.path.join(IMAGE_DIR, dataset_type)
  os.makedirs(dst_dir, exist_ok=True)
  seq_dir = '/content'

  obj_dict = {}

  last_track_id = 0
  max_track_id = 0

  wh = []

  dataset_dicts = []

  seq_dirs = [seq['dir'] for seq in sequences[dataset_type] if seq['type']=="full"]

  for seq_id, seq in enumerate(seq_dirs):

    ann_dir = os.path.join(seq_dir, seq[0], seq[1], 'ann')
    img_dir = os.path.join(seq_dir, seq[0], seq[1], 'img')

    for idx, ann in enumerate(sorted(os.listdir(os.path.join(seq_dir, seq[0], seq[1], 'ann')))):

      with open(os.path.join(ann_dir, ann)) as f:
        ann_dict = json.load(f)

      img_file = ann[:-5]
      image = cv2.imread(os.path.join(img_dir, img_file))

      bboxes_dict = {}

      for obj_idx, obj in enumerate(ann_dict["objects"]):

        if obj["classTitle"] in ["player"]: # TODO: może szerzej?
          bbox_ext = obj["points"]["exterior"]
          x,y,w,h = cv2.boundingRect(np.array(bbox_ext))
          bbox_ext = [x, y, x+w, y+h]

          #area_ext = (bbox_ext[2]-bbox_ext[0])*(bbox_ext[3]-bbox_ext[1])

          bboxes = []
          for obj_in in ann_dict["objects"]:
            if obj_in["classTitle"] == "shirt-number":
              bbox = obj_in["points"]["exterior"]
              bbox = bbox[0]+bbox[1]
              bbox = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])]

              #area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])              

              if bbox_ext[0]<=bbox[0] and bbox_ext[1]<=bbox[1] and bbox_ext[2]>=bbox[2] and bbox_ext[3]>=bbox[3]: # TODO: a co jak trochę wystaje?
                bboxes.append(bbox)

          if not bboxes:
            continue

          crop_file = f"{img_file[:-4]}_{obj_idx}{img_file[-4:]}"
          crop = image[bbox_ext[1]:bbox_ext[3], bbox_ext[0]:bbox_ext[2]]
          cv2.imwrite(os.path.join(dst_dir, crop_file), crop)

          record = {}
          record["file_name"] = os.path.join(dst_dir, crop_file)
          record["image_id"] = 10000*seq_id + 100*idx + obj_idx
          record["height"] = crop.shape[0]
          record["width"] = crop.shape[1]
          record["annotations"] = []

          for bbox in bboxes:
            bbox[0] -= bbox_ext[0]
            bbox[1] -= bbox_ext[1]
            bbox[2] -= bbox_ext[0]
            bbox[3] -= bbox_ext[1]
          
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
                "iscrowd": 0
            }
            record["annotations"].append(obj)
          dataset_dicts.append(record)         

  return dataset_dicts

def get_supervisely_dicts(base_dir):

  ann_dir = os.path.join(base_dir, 'ann')
  img_dir = os.path.join(CROP_SRC_DIR, 'bbox_crops') #os.path.join(base_dir, 'img') #IMG_DIR #os.path.join(base_dir, 'img')

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
      if anno["classTitle"] == "shirt-number":
        bbox = anno["points"]["exterior"]
        bbox = bbox[0]+bbox[1]
        bbox = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])]
        #bbox = [min(bbox[0], bbox[2])*record["width"]/224, min(bbox[1], bbox[3])*record["height"]/384, max(bbox[0], bbox[2])*record["width"]/224, max(bbox[1], bbox[3])*record["height"]/384] 
        obj = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0,
            "iscrowd": 0
        }
        objs.append(obj)
    record["annotations"] = objs
    if len(objs)>0:
      dataset_dicts.append(record)
  return dataset_dicts

# todo: poprawić
def get_seq_dicts(d, seq_dir, sequences):
  dataset_dicts = []

  crop_seqs = [seq['dir'] for seq in sequences[d] if seq['type']=="crop"]
  for seq in crop_seqs:
    dataset_dicts.extend(get_supervisely_dicts(os.path.join(seq_dir, seq[0], seq[1])))
  
  dataset_dicts.extend(get_supervisely_dicts_cropped(sequences, d))
  return dataset_dicts

def register_number_instances():
  seq_dir = '/content'

  DatasetCatalog.clear()
  for d in ["train", "val"]:
      DatasetCatalog.register("supervisely_" + d, lambda d=d: get_seq_dicts(d, seq_dir, SEQUENCES))
      MetadataCatalog.get("supervisely_" + d).set(thing_classes=["shirt_number"])
  supervisely_metadata = MetadataCatalog.get("supervisely_train")
