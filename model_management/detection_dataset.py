import os

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from model_management.detection_annotations import extract_boxes_and_labels, load_annotations
from model_management.detection_transforms import Compose, ToTensor


class DetectionDataset(Dataset):
    def __init__(self, frames, transforms=None):
        if transforms is None:
            transforms = Compose((ToTensor(),))
        self.frames = frames
        self.transforms = transforms

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame_info = self.frames[index]
        img = Image.open(frame_info["path"]).convert("RGB")
        target = {
            "boxes": torch.as_tensor(frame_info["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(frame_info["labels"], dtype=torch.int64),
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


class TrafficDataset(DetectionDataset):
    def __init__(self, root=None, select_index=None):
        self.root = root
        frames = collect_frames(root, select_index)
        super(TrafficDataset, self).__init__(frames)


def collect_frames(root, select_index):
    frames = []
    frame_path = os.path.join(root, "frames")
    frame_names = list(os.listdir(frame_path))

    annotation_path = os.path.join(root, "annotation.txt")
    annotations = load_annotations(annotation_path)
    logger.debug(frames)
    for frame_name in frame_names:
        _id = int(frame_name.split(".")[0])
        logger.debug("id {} name{}, select{}".format(_id, frame_name, select_index))
        if _id in select_index:
            logger.debug(_id)
            _path = os.path.join(frame_path, frame_name)
            _labels = annotations[annotations["frame_index"] == _id]
            boxes, labels = extract_boxes_and_labels(_labels)

            if len(boxes) != 0 and len(labels) != 0:
                frames.append({"path": _path, "frame_index": _id, "boxes": boxes, "labels": labels})

    return frames
