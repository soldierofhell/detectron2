import argparse
import os
import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog

from vovnet import add_vovnet_config
from point_rend import add_pointrend_config

#import vovnet
#import point_rend

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

def create_players_coco(image_dir, gt_json, predicted_json, cfg_path, mask_on, num_classes, thing_classes, vovnet, checkpoint, nms_threshold, detection_threshold, batch_size):
  
  dataset_name = "inference_dataset" # todo: randomize?
  
  DatasetCatalog.clear()
  
  if gt_json is not None:
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances(dataset_name, {}, gt_json, image_root)
  else:
    DatasetCatalog.register(dataset_name, lambda: get_img_dicts(image_dir))    
    MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)
  
  print('classes: ', thing_classes)

  cfg = get_cfg()
  
  if vovnet:
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    
    #from point_rend.config import add_pointrend_config
    #from vovnet import add_vovnet_config
    add_vovnet_config(cfg)
    add_pointrend_config(cfg)

  # Cascade Mask RCNN ResNeXt 152
  cfg.merge_from_file(cfg_path)
  cfg.MODEL.WEIGHTS = checkpoint
  cfg.MODEL.MASK_ON = False
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes #1
  cfg.DATALOADER.NUM_WORKERS = batch_size
  


  model = build_model(cfg)
  model.eval()
  DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

  output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

  coco_data_loader = build_detection_test_loader(cfg, dataset_name) # batch_size = cfg.DATALOADER.NUM_WORKERS
  coco_evaluator = COCOEvaluator(dataset_name, cfg, False, output_folder)

  # TODO: if gt_json is not None: don't calculate
  
  inference_on_dataset(model, coco_data_loader, evaluator=coco_evaluator)  

  with open(os.path.join(output_folder, 'inference_dataset_coco_format.json'), 'r') as f:
    coco_dict = json.load(f)

  with open(os.path.join(output_folder, 'coco_instances_results.json'), 'r') as f:
    annotations_list = json.load(f)

  for idx, ann in enumerate(annotations_list):
    ann['id'] = idx
    ann['bbox'] = [int(x) for x in ann['bbox']]

  coco_dict['annotations'] = annotations_list

  with open(predicted_json, 'w') as f:
    json.dump(coco_dict, f)
  
def parse_args():
  parser = argparse.ArgumentParser()  
  parser.add_argument("--image_dir", type=str)
  parser.add_argument("--gt_json", type=str)
  parser.add_argument("--cfg", type=str, default="./detectron2_repo/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
  parser.add_argument("--checkpoint", type=str, default="detectron2://Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/model_0039999_e76410.pkl")
  parser.add_argument("--predicted_json", type=str)
  parser.add_argument("--nms_threshold", type=float, default=0.5)
  parser.add_argument("--detection_threshold", type=float, default=0.5)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--vovnet", action="store_true")
  parser.add_argument("--mask_on", action="store_true")
  parser.add_argument("--num_classes", type=int, default=1)
  parser.add_argument('--thing_classes', nargs='+')
  
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  create_players_coco(args.image_dir, args.gt_json, args.predicted_json, args.cfg, args.mask_on, args.num_classes, args.thing_classes, args.vovnet, args.checkpoint, args.nms_threshold, args.detection_threshold, args.batch_size)
