from collections.abc import Mapping, Sequence

import numpy as np
from mapcalc import calculate_map
from ultralytics.utils.plotting import Annotator, colors

from model_management.model_info import COCO_INSTANCE_CATEGORY_NAMES


def cal_iou(a, b):
    a_x1 = a[0]
    a_y1 = a[1]
    a_x2 = a[2]
    a_y2 = a[3]

    b_x1 = b[0]
    b_y1 = b[1]
    b_x2 = b[2]
    b_y2 = b[3]

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection_area = inter_w * inter_h

    a_w = max(0, a_x2 - a_x1)
    a_h = max(0, a_y2 - a_y1)
    a_area = a_w * a_h

    b_w = max(0, b_x2 - b_x1)
    b_h = max(0, b_y2 - b_y1)
    b_area = b_w * b_h

    union_area = a_area + b_area - intersection_area
    return intersection_area / union_area


def get_offloading_region(high_detections, low_regions, img_shape):
    """
    Remove larger areas
    Remove overlapping areas
    :param detection:
    :param inferencer:
    :return:
    """
    offloading_region = []

    img_height = img_shape[0]
    img_width = img_shape[1]

    for i in range(len(low_regions)):
        x1 = low_regions[i][0]
        y1 = low_regions[i][1]
        x2 = low_regions[i][2]
        y2 = low_regions[i][3]
        if (x2 - x1) * (y2 - y1) > img_width * img_height * 0.1:
            continue
        if high_detections is not None:
            match = 0
            for j in range(len(high_detections)):
                if cal_iou(low_regions[i], high_detections[j]) > 0.3:
                    match += 1
            if match > 0:
                continue
        offloading_region.append(low_regions[i])
    return offloading_region


def get_offloading_image(offloading_region, image):
    """
    Put these areas on a black background and encode them with high quality
    :param region_query:
    :return:
    """
    cropped_image = np.zeros_like(image)
    width = image.shape[1]
    hight = image.shape[0]
    cached_image = image
    for i in range(len(offloading_region)):
        x1 = int(offloading_region[i][0]) - 1 if int(offloading_region[i][0]) - 1 >= 0 else 0
        y1 = int(offloading_region[i][1]) - 1 if int(offloading_region[i][1]) - 1 >= 0 else 0
        x2 = (
            int(offloading_region[i][2]) + 1 if int(offloading_region[i][2]) + 1 <= width else width
        )
        y2 = (
            int(offloading_region[i][3]) + 1 if int(offloading_region[i][3]) + 1 <= hight else hight
        )
        cropped_image[y1:y2, x1:x2, :] = cached_image[y1:y2, x1:x2, :]
    return cropped_image


def _clip_box_to_image(box, image_shape):
    if box is None or len(box) < 4:
        return None

    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        return None

    x1 = int(round(float(box[0])))
    y1 = int(round(float(box[1])))
    x2 = int(round(float(box[2])))
    y2 = int(round(float(box[3])))

    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, 0), width - 1)
    y2 = min(max(y2, 0), height - 1)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _is_zero_based_label_schema(label_schema) -> bool:
    return str(label_schema or "").strip().lower() == "zero_based"


def _resolve_class_name(label_index: int, class_names, label_schema=None):
    if not class_names:
        return None

    if isinstance(class_names, Mapping):
        value = class_names.get(label_index, class_names.get(str(label_index)))
        return None if value is None else str(value)

    if isinstance(class_names, Sequence) and not isinstance(class_names, (str, bytes)):
        if _is_zero_based_label_schema(label_schema):
            if 0 <= label_index < len(class_names):
                return str(class_names[label_index])
            return None

        if 1 <= label_index <= len(class_names):
            return str(class_names[label_index - 1])
        if 0 <= label_index < len(class_names):
            return str(class_names[label_index])

    return None


def _resolve_label_text(label, class_names=None, label_schema=None):
    if isinstance(label, (int, np.integer)):
        label_index = int(label)
        class_name = _resolve_class_name(label_index, class_names, label_schema)
        if class_name is not None:
            return class_name
        if _is_zero_based_label_schema(label_schema):
            return f"class_{label_index}"
        return (
            COCO_INSTANCE_CATEGORY_NAMES[label_index]
            if 0 <= label_index < len(COCO_INSTANCE_CATEGORY_NAMES)
            else str(label_index)
        )
    return str(label)


def _resolve_color_index(label, label_text):
    if isinstance(label, (int, np.integer)):
        return int(label)
    if label_text in COCO_INSTANCE_CATEGORY_NAMES:
        return COCO_INSTANCE_CATEGORY_NAMES.index(label_text)
    return sum(ord(char) for char in label_text) % 20


def _resolve_annotation_line_width(image_shape):
    if image_shape is None or len(image_shape) < 2:
        return 1
    min_dim = min(int(image_shape[0]), int(image_shape[1]))
    return max(1, min(2, int(round(min_dim / 1400.0))))


def draw_detection(img, pred_boxes, pred_cls, pred_score, class_names=None, label_schema=None):
    cached_img = np.ascontiguousarray(img.copy())
    if pred_boxes is None or pred_cls is None:
        return cached_img
    annotator = Annotator(
        cached_img,
        pil=False,
        example="abc",
        line_width=_resolve_annotation_line_width(cached_img.shape),
    )
    for i in range(len(pred_boxes)):
        if i >= len(pred_cls):
            break
        clipped_box = _clip_box_to_image(pred_boxes[i], cached_img.shape)
        if clipped_box is None:
            continue
        label = pred_cls[i]
        label_text = _resolve_label_text(
            label,
            class_names=class_names,
            label_schema=label_schema,
        )

        score_text = ""
        if pred_score is not None and i < len(pred_score):
            score_text = f" {float(pred_score[i]):.2f}"

        annotator.box_label(
            clipped_box,
            f"{label_text}{score_text}",
            color=colors(_resolve_color_index(label, label_text), True),
        )
    return annotator.result()


def cal_mAP(ground_truth, result_dict):
    # calculates the mAP for an IOU threshold of 0.5
    return calculate_map(ground_truth, result_dict, 0.5)


if __name__ == "__main__":
    pass
