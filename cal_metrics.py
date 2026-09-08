import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import numpy as np
from loguru import logger
from mapcalc import calculate_map

from config import load_runtime_config
from model_management.object_detection import Object_Detection
from tools.video_processor import VideoProcessor


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class CalMetrics:
    def __init__(self, config, result_path: str):
        self.client_config = config.client
        self.source = config.client.source
        self.inference_config = config.server
        self.result_path = Path(result_path)
        self.large_object_detection = Object_Detection(
            self.inference_config,
            type="large inference",
        )

    def cal_ground_truth(self):
        with VideoProcessor(self.source) as video:
            truth_dict = {}
            video_fps = video.fps
            interval = self.client_config.interval
            logger.info("the video fps is {}", video_fps)
            if interval == 0:
                raise ValueError("client.interval must not be 0")

            logger.info("Take the frame interval is {}", interval)
            index = 0
            while True:
                frame = next(video)
                if frame is None:
                    logger.info("the video is over")
                    break
                index += 1
                if index % interval == 0:
                    truth_boxes, truth_class, _ = self.large_object_detection.large_inference(frame)
                    truth_dict[str(index)] = {
                        "labels": truth_class,
                        "boxes": truth_boxes,
                    }

        with open("truth.json", "w", encoding="utf-8") as f:
            json.dump(truth_dict, f, indent=4, cls=Encoder)

    def _load_results(self) -> list[dict]:
        if not self.result_path.exists():
            raise FileNotFoundError(f"Result file not found: {self.result_path}")

        results = []
        with self.result_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                results.append(json.loads(line))
        return results

    def cal_mAP(self):
        with open("truth.json", "r", encoding="utf-8") as f:
            ground_truths = json.load(f)

        results = self._load_results()
        if not results:
            logger.warning("No inference results found in {}", self.result_path)
            return {"map": 0.0, "delay": 0.0, "frames": 0}

        sum_map = 0.0
        sum_delay = 0.0
        evaluated = 0

        for record in results:
            frame_index = int(record["frame_index"])
            ground_truth = ground_truths.get(str(frame_index))
            if ground_truth is None:
                logger.warning("Missing ground truth for frame {}", frame_index)
                continue

            prediction = record.get("result") or {
                "labels": [],
                "boxes": [],
                "scores": [],
            }
            sum_map += calculate_map(ground_truth, prediction, 0.5)

            start_time = record.get("start_time")
            end_time = record.get("end_time")
            if start_time is not None and end_time is not None:
                sum_delay += float(end_time) - float(start_time)

            evaluated += 1

        avg_map = sum_map / evaluated if evaluated else 0.0
        avg_delay = sum_delay / evaluated if evaluated else 0.0

        logger.info("Evaluated {} frames from {}", evaluated, self.result_path)
        logger.info("mAP@0.5 = {:.4f}", avg_map)
        logger.info("Average delay = {:.4f} s", avg_delay)
        return {"map": avg_map, "delay": avg_delay, "frames": evaluated}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument(
        "--yaml_path", default="./config/config.yaml", help="input the path of *.yaml"
    )
    parser.add_argument(
        "--result_path",
        default="./log/client/latest_inference_results.jsonl",
        help="path to the local inference result log",
    )
    parser.add_argument(
        "--cal_truth",
        action="store_true",
        help="recompute truth.json with the server model before evaluating",
    )
    args = parser.parse_args()

    config = load_runtime_config(args.yaml_path)
    cal = CalMetrics(config, args.result_path)
    if args.cal_truth:
        cal.cal_ground_truth()
    cal.cal_mAP()
