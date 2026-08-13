import csv
from dataclasses import dataclass
from itertools import islice
from math import fabs
import os
from pathlib import Path
import random
import sys
import shutil

from PIL import Image


# Label map adapted from source dataset ReadMe.txt
label_txt = {
  0: "speed_limit_20",
  1: "speed_limit_30",
  2: "speed_limit_50",
  3: "speed_limit_60",
  4: "speed_limit_70",
  5: "speed_limit_80",
  6: "restriction_ends_80",
  7: "speed_limit_100",
  8: "speed_limit_120",
  9: "no_overtaking",
  10: "no_overtaking_trucks",
  11: "priority_at_next_intersection",
  12: "priority_road",
  13: "give_way",
  14: "stop",
  15: "no_traffic_both_ways",
  16: "no_trucks",
  17: "no_entry",
  18: "danger",
  19: "bend_left",
  20: "bend_right",
  21: "bend",
  22: "uneven_road",
  23: "slippery_road",
  24: "road_narrows",
  25: "construction",
  26: "traffic_signal",
  27: "pedestrian_crossing",
  28: "school_crossing",
  29: "cycles_crossing",
  30: "snow",
  31: "animals",
  32: "restriction_ends",
  33: "go_right",
  34: "go_left",
  35: "go_straight",
  36: "go_right_or_straight",
  37: "go_left_or_straight",
  38: "keep_right",
  39: "keep_left",
  40: "roundabout",
  41: "restriction_ends_overtaking",
  42: "restriction_ends_overtaking_trucks"
}


dataset_yaml_template = """
path: dataset

train: train/images
val: val/images
test: test/images

# Class count
nc: 43

names:
  0: speed_limit_20
  1: speed_limit_30
  2: speed_limit_50
  3: speed_limit_60
  4: speed_limit_70
  5: speed_limit_80
  6: restriction_ends_80
  7: speed_limit_100
  8: speed_limit_120
  9: no_overtaking
  10: no_overtaking_trucks
  11: priority_at_next_intersection
  12: priority_road
  13: give_way
  14: stop
  15: no_traffic_both_ways
  16: no_trucks
  17: no_entry
  18: danger
  19: bend_left
  20: bend_right
  21: bend
  22: uneven_road
  23: slippery_road
  24: road_narrows
  25: construction
  26: traffic_signal
  27: pedestrian_crossing
  28: school_crossing
  29: cycles_crossing
  30: snow
  31: animals
  32: restriction_ends
  33: go_right
  34: go_left
  35: go_straight
  36: go_right_or_straight
  37: go_left_or_straight
  38: keep_right
  39: keep_left
  40: roundabout
  41: restriction_ends_overtaking
  42: restriction_ends_overtaking_truck
"""


@dataclass
class Label:
  path: str
  from_x: int
  from_y: int
  to_x: int
  to_y: int
  id: int

  img_width = 1360
  img_height = 800

  def width(self):
    return (self.to_x - self.from_x) / self.img_width

  def height(self):
    return (self.to_y - self.from_y) / self.img_height

  def center_x(self):
    return self.from_x / self.img_width + self.width() / 2

  def center_y(self):
    return self.from_y / self.img_height + self.height() / 2


def parse_arg():
  if len(sys.argv) < 3:
    print("Please specify the source and destination of the dataset")
    exit(1)

  return sys.argv[1], sys.argv[2]


def create_base_structure(dest: str):
  # Remove dest folder if exists
  path = Path(dest)
  # if path.exists() and path.is_dir():
  #   shutil.rmtree(path)

  # Recreate folder structure
  path.mkdir(parents=True, exist_ok=True)

  for type in [ "train", "val", "test" ]:
    Path(dest, type, "images").mkdir(parents=True, exist_ok=True)
    Path(dest, type, "labels").mkdir(parents=True, exist_ok=True)


def load_labels(src: str):
  ppm_imgs = [f.name for f in Path(src).glob('*.ppm')]
  labels: list[Label] = list()
  imgs: dict[str, list[Label]] = dict()

  with open(os.path.join(src, "gt.txt"), newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    for row in spamreader:
      label = Label(
        row[0],
        int(row[1]),
        int(row[2]),
        int(row[3]),
        int(row[4]),
        int(row[5]),
      )
      labels.append(label)

  for ppm_img in ppm_imgs:
    imgs[ppm_img] = [ x for x in labels if x.path == ppm_img ]

  return imgs


def split_labels(labels, train, val, test, randomize):
  # if randomize:
  #   labels = dict(labels)
  #   random.shuffle(labels)

  train_cnt = int(len(labels) * train)
  val_cnt = int(len(labels) * val)

  return dict(islice(labels.items(), 0, train_cnt)), dict(islice(labels.items(), train_cnt, train_cnt+val_cnt)), dict(islice(labels.items(), train_cnt+val_cnt, None))


def gen_yolo_label_file(labels: list[Label]):
  yolo_labels = ""

  for label in labels:
    # yolo_labels += f"{label.id} {label.center_x()} {label.center_y()} {label.width()} {label.height()} {label.from_x} {label.from_y} {label.to_x} {label.to_y} {label.width() / 1360} {label.center_x() / 1360}\n"
    yolo_labels += f"{label.id} {label.center_x()} {label.center_y()} {label.width()} {label.height()}\n"

  return yolo_labels


def write_yolo_data(src: str, dest: str, labels_train: dict[str, list[Label]], labels_val: dict[str, list[Label]], labels_test: dict[str, list[Label]]):
  cycles = {
    "train": labels_train,
    "val": labels_val,
    "test": labels_test,
  }

  for cycle, img_labels in cycles.items():
    for img, labels in img_labels.items():
      print(f"Converting {img}")

      yolo_labels = gen_yolo_label_file(labels)

      # shutil.copyfile(Path(src, img), Path(dest, cycle, "images", img))

      # Convert ppm to png
      with Image.open(Path(src, img)) as i:
        i.save(Path(dest, cycle, "images", img.replace(".ppm", ".png")), "PNG")

      # Write yolo labels
      with open(Path(dest, cycle, "labels", img.replace(".ppm", ".txt")), "w") as f:
        f.write(yolo_labels)


def write_dataset_yaml():
  with open("dataset.yml", "w") as f:
    f.write(dataset_yaml_template)


src = "FullIJCNN2013"
dest = "dataset"
randomize_labels = False
train = 0.7
val = 0.15
test = 0.15

img_labels = load_labels(src)

img_labels_train, img_labels_val, img_labels_test = split_labels(img_labels, train, val, test, randomize_labels)

#print(img_labels_train, len(img_labels_train), len(img_labels_val), len(img_labels_test), len(img_labels_train) + len(img_labels_val) + len(img_labels_test), len(img_labels))
# print(img_labels.keys())

create_base_structure(dest)
write_yolo_data(src, dest, img_labels_train, img_labels_val, img_labels_test)
write_dataset_yaml()
