import torch
import random
from torchvision import datasets, transforms
from src.models import RetinaNet
from src.utils import *


CONTAINER_PATH = "/mnt/vlgrounding" # where the data is mounted to
COCO_MAPPING = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def get_labels_to_id(name):
    if name == "coco":
        return {v: i for i, v in enumerate(COCO_MAPPING.values())}
    if name == "voc":
        return {v: i for i, v in enumerate(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"])}

def get_id_to_labels(name):
    return {v: k for k, v in get_labels_to_id(name).items()}

def get_num_labels(name):
    return len(get_labels_to_id(name))

def get_normalization_stats():
    """
    Normalization stats taken from ImageNet, shared across all datasets.
    """
    return {"mu": [0.485, 0.456, 0.406], "sigma": [0.229, 0.224, 0.225]}

def unnormalize(x):
    """
    For plotting purposes, unnormalize an image represented as a tensor using ImageNet stats.
    """
    normalization = get_normalization_stats()
    sigma = torch.tensor(normalization["sigma"]).unsqueeze(1).unsqueeze(2)
    mu = torch.tensor(normalization["mu"]).unsqueeze(1).unsqueeze(2)
    return x * sigma + mu

def extract_labels_boxes_from_annotation(annotation, name):
    """
    Dataset specific munging to clean up annotations for datasets into a consistent representation.

    Returns
    -------
    labels: variable length list of integers representing labels of each box
    boxes: variable length list of tuples of 4 floats (ymin, xmin, ymax, xmax) of each box
    """
    if name == "voc":

        labels_to_id = get_labels_to_id(name)
        bbox_fn = lambda d: (float(d["ymin"]), float(d["xmin"]), float(d["ymax"]), float(d["xmax"]))
        if isinstance(annotation["annotation"]["object"], list):
            labels = [labels_to_id[obj["name"]] for obj in annotation["annotation"]["object"]]
            boxes = [bbox_fn(obj["bndbox"]) for obj in annotation["annotation"]["object"]]
        else:
            labels = [labels_to_id[annotation["annotation"]["object"]["name"]]]
            boxes = [bbox_fn(annotation["annotation"]["object"]["bndbox"])]
        return labels, boxes

    if name == "coco":

        labels_to_id = get_labels_to_id(name)
        bbox_fn = lambda t: (t[1], t[0], t[1] + t[3], t[0] + t[2])
        labels = [labels_to_id[COCO_MAPPING[obj["category_id"]]] for obj in annotation]
        boxes = [bbox_fn(obj["bbox"]) for obj in annotation]
        return labels, boxes

    raise ValueError

def get_dataset(name, split):
    
    stats = get_normalization_stats()
    target_transform = lambda annotation: extract_labels_boxes_from_annotation(annotation, name)

    if name == "voc" and split == "train":
        return datasets.VOCDetection("./data/voc", image_set="train", download=True, 
                                     year="2012",
                                     transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize(stats["mu"], stats["sigma"])]),
                                     target_transform=target_transform)

    if name == "voc" and split == "test":
        return datasets.VOCDetection("./data/voc", image_set="val", download=True, 
                                     year="2012",
                                     transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize(stats["mu"], stats["sigma"])]),
                                     target_transform=target_transform)
                                            
    
    if name == "coco" and split == "train":
        return datasets.CocoDetection(f"{CONTAINER_PATH}/coco/train2014", 
                                      f"{CONTAINER_PATH}/coco/annotations/instances_train2014.json",
                                      transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(stats["mu"], stats["sigma"])]),
                                      target_transform=target_transform)

    if name == "coco" and split == "test":
        return datasets.CocoDetection(f"{CONTAINER_PATH}/coco/val2014",
                                      f"{CONTAINER_PATH}/coco/annotations/instances_val2014.json",
                                      transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(stats["mu"], stats["sigma"])]),
                                      target_transform=target_transform)

    raise ValueError


def collate_fn(batch, name):
    """
    Function to convert a raw batch into encoded targets for RetinaNet.

    Notes
    -----
    Images in the batch are zero-padded to the maximum width and height.
    Note that  PyTorch dataloaders default order (channel, height, width); this must be respected
    to maintain consistency with pre-trained weights.

    Returns
    -------
    images: (batch_size, num_classes, width, height) tensor of padded inputs
    cls_tgts: (# fm)-list of tensors each of size (batch_size, num_classes, num_anchors, fmw, fmh)
    reg_tgts: (# fm)-list of tensors each of size (batch_size, 4, num_anchors, fmw, fmh)
    """
    batch_size = len(batch)
    max_height = max([x.shape[1] for (x, y) in batch])
    max_width = max([x.shape[2] for (x, y) in batch])

    images = torch.zeros((batch_size, 3, max_height, max_width), dtype=torch.float)
    cls_tgts = [[] for _ in range(RetinaNet.num_feature_maps)]
    reg_tgts = [[] for _ in range(RetinaNet.num_feature_maps)]
    num_labels = get_num_labels(name)

    for i, (x, annotation) in enumerate(batch):

        height, width = x.shape[1], x.shape[2]
        images[i, :, :height, :width] = x
        labels, boxes = annotation
        boxes = [torch.tensor(bbox_convert_corners_to_sizes(b), dtype=torch.float) for b in boxes]
        cls_tgts_i, reg_tgts_i = RetinaNet.encode(images[i], labels, boxes, num_labels)

        for fm_no in range(RetinaNet.num_feature_maps):
            cls_tgts[fm_no].append(cls_tgts_i[fm_no])
            reg_tgts[fm_no].append(reg_tgts_i[fm_no])

    for fm_no in range(RetinaNet.num_feature_maps):
        cls_tgts[fm_no] = torch.stack(cls_tgts[fm_no])
        reg_tgts[fm_no] = torch.stack(reg_tgts[fm_no])

    return images, cls_tgts, reg_tgts
   
