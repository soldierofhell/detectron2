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

import json
import os

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.data import build_detection_test_loader

def create_players_coco(image_dir, json_path, checkpoint, nms_threshold, detection_threshold, batch_size):
  DatasetCatalog.clear()
  DatasetCatalog.register("inference_dataset", lambda: get_img_dicts('/content/LhcTGafSgw_all'))
  MetadataCatalog.get("inference_dataset").set(thing_classes=["person"])

  cfg = get_cfg()

  # Cascade Mask RCNN ResNeXt 152
  cfg.merge_from_file("./detectron2_repo/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
  cfg.MODEL.WEIGHTS = checkpoint
  cfg.MODEL.MASK_ON = False
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
  cfg.DATALOADER.NUM_WORKERS = batch_size

  model = build_model(cfg)
  model.eval()
  DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

  val_dataset = "inference_dataset"
  output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

  coco_data_loader = build_detection_test_loader(cfg, val_dataset) # batch_size = cfg.DATALOADER.NUM_WORKERS
  coco_evaluator = COCOEvaluator(val_dataset, cfg, False, output_folder)

  inference_on_dataset(model, coco_data_loader, evaluator=coco_evaluator)  

  with open(os.path.join(output_folder, 'inference_dataset_coco_format.json'), 'r') as f:
    coco_dict = json.load(f)

  with open(os.path.join(output_folder, 'coco_instances_results.json'), 'r') as f:
    annotations_list = json.load(f)

  for idx, ann in enumerate(annotations_list):
    ann['id'] = idx

  coco_dict['annotations'] = annotations_list

  with open(json_path, 'w') as f:
    json.dump(coco_dict, f)
  
def parse_args():
  parser = argparse.ArgumentParser()  
  parser.add_argument("--image_dir, type=str)
  parser.add_argument("--json_path, type=str)
  parser.add_argument("--checkpoint", type=str, default="detectron2://Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/model_0039999_e76410.pkl")
  parser.add_argument("--nms_threshold, type=float, default=0.5)
  parser.add_argument("--detection_threshold, type=float, default=0.5)
  parser.add_argument("--batch_size, type=int, default=1)
  
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  create_players_coco(args.image_dir, args.json_path, args.checkpoint, args.nms_threshold, args.detection_threshold, args.batch_size)