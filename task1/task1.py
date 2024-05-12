# import some common libraries
import numpy as np
import shutil, os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def detect_person(img):
  # return predictions of class 'person' in the image
  person_id = THING_CLASS_NAMES.index('person')
  predictions = detect(img)
  return predictions['instances'][predictions['instances'].pred_classes == person_id].to("cpu")


def segment_sea(img):
  # segment sea in the image and return sea mask. Classes sea/water/river are considered 'sea'
  sea_id = STUFF_CLASS_NAMES.index('sea')
  water_id = STUFF_CLASS_NAMES.index('water')
  river_id = STUFF_CLASS_NAMES.index('river')
  predictions = segment(img)
  sem_mask = predictions["sem_seg"].argmax(dim=0).to("cpu").numpy()
  mask = np.uint8((sem_mask == sea_id) | (sem_mask == water_id) | (sem_mask == river_id))
  return mask


def segment_sky(img):
  # segment sky in the image and return sky mask
  sky_id = STUFF_CLASS_NAMES.index('sky')
  predictions = segment(img)
  sem_mask = predictions["sem_seg"].argmax(dim=0).to("cpu").numpy()
  mask = np.uint8(sem_mask == sky_id)
  return mask


def extract_large_polygons(mask, min_ratio=0.5):
  # extract largest polygon and the next one if it is at least half (set by min_ratio) of the largest one
  polygons, hir = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  polygons = sorted(polygons, key=cv2.contourArea, reverse=True)
  polygons = [polygon.squeeze(axis=1) for polygon in polygons]
  if len(polygons) >= 2 and cv2.contourArea(polygons[1]) >= min_ratio*cv2.contourArea(polygons[0]):
      return polygons[:2]
  return polygons[:1]


def sky_reaches_bottom(img, polygons):
  # heuristics used to detect sky reflection in the sea
  ymax_img = img.shape[0]
  for polygon in polygons:
    ymax_poly = polygon[:, 1].max()
    if ymax_poly == ymax_img - 1:
      return True
  return False


def bbox_in_poly(bbox, polygons):
  # check if the center of bbox is in one of polygons
  x1, y1, x2, y2 = bbox
  center_of_mass = (int((x2 + x1) / 2), int((y2 + y1) / 2))
  for polygon in polygons:
    is_in_poly = cv2.pointPolygonTest(polygon, center_of_mass, measureDist=False)
    if is_in_poly >= 1:
      return True
  return False


def person_in_sea(person_instances, sea_polygons):
  # return true if there is a person in the sea
  for bbox in person_instances.pred_boxes:
    if bbox_in_poly(bbox, sea_polygons):
      return True
  return False


def draw_instances(img, predictions):
  v = Visualizer(img[:, :, ::-1])
  out = v.draw_instance_predictions(predictions)
  return out.get_image()[:, :, ::-1]


def draw_polygons(img, polygons, color):
  if len(polygons) == 0:
    return img
  v = Visualizer(img[:, :, ::-1])
  for polygon in polygons:
    out = v.draw_polygon(polygon, color, edge_color=color, alpha=0.2)
  return out.get_image()[:, :, ::-1]


###############################################################################
# setup detection
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"
THING_CLASS_NAMES = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
detect = DefaultPredictor(cfg)

# setup segmentation
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
cfg.MODEL.DEVICE = "cpu"
STUFF_CLASS_NAMES = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes
segment = DefaultPredictor(cfg)

# define paths
dir = {
    "in": "./data",
    "out_with": "./out_data_with",
    "out_without": "./out_data_without",
    "viz": "./viz_data"
}

for k in ["out_with", "out_without", "viz"]:
  os.makedirs(dir[k], exist_ok=True)

# main loop
filenames = sorted(os.listdir(dir["in"]))
for filename in filenames:
  filepath = {k: os.path.join(dir[k], filename) for k in dir.keys()}
  img = cv2.imread(filepath["in"])
  person_instances = detect_person(img)
  sea_mask = segment_sea(img)
  sea_polygons = extract_large_polygons(sea_mask)
  print(f'{filename}: {len(person_instances)} person instances, {len(sea_polygons)} sea segments')
  sky_mask = segment_sky(img)
  sky_polygons = extract_large_polygons(sky_mask)
  if sky_reaches_bottom(img, sky_polygons):
    print(f'{filename}: sky reflection in the sea')
    sea_polygons = sky_polygons
  if person_in_sea(person_instances, sea_polygons):
    print(f'{filename}: there is a person in the sea')
    shutil.copy(filepath["in"], filepath["out_with"])
  else:
    print(f'{filename}: there is no person in the sea')
    shutil.copy(filepath["in"], filepath["out_without"])

  # vizualizations
  img = draw_instances(img, person_instances)
  img = draw_polygons(img, sea_polygons, "blue")
  img = draw_polygons(img, sky_polygons, "cyan")
  cv2.imwrite(filepath["viz"], img)
