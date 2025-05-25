import logging
import os
from typing import Any

import pandas as pd
import torch
from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh
from ultralytics import YOLO
import numpy as np

log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOUltralytics(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
        "yolo_class",  # Store original YOLO class index for the fine-tuned model
    ]

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.model = YOLO(cfg.path_to_checkpoint)
        self.model.to(device)
        self.id = 0

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images, shapes = batch
        results_by_image = self.model(images, verbose=False)
        detections = []

        # Define class mapping (YOLO class index → category_id in pipeline)
        class_mapping = {
            0: 1,  # player
            1: 1,  # goalkeeper 
            2: 1,  # referee 
            3: 2,  # ball
            4: 1,  # other 
        }

        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in results.boxes.cpu().numpy():
                # Extract the class as an integer (not an array)
                class_idx = int(bbox.cls[0]) if isinstance(bbox.cls, (list, np.ndarray)) else int(bbox.cls)
                
                # check for `person, player, goalkeeper, referee and other` class
                if class_idx in [0, 1, 2, 4] and bbox.conf >= self.cfg.min_confidence:  # person classes
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
                                bbox_conf=bbox.conf[0],
                                video_id=metadata.video_id,
                                category_id=class_mapping[class_idx],  # Use the integer class index
                                yolo_class=class_idx,  # Store original class for later use
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections