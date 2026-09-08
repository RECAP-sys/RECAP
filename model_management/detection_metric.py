import time

from numpy import average as avg
from tqdm import tqdm


class RetrainMetric:
    DEFAULT_LOSS_KEYS = (
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    )

    def __init__(self):
        self.metrics = {}

    def reset_metrics(self):
        self.metrics = {key: [] for key in self.DEFAULT_LOSS_KEYS}
        self.metrics.update(
            {
                "total_loss": [],
            }
        )

    def update(self, loss_dict, total_loss):
        for loss_name, loss_value in loss_dict.items():
            self.metrics.setdefault(loss_name, [])
            self.metrics[loss_name].append(loss_value.detach().cpu().item())
        self.metrics["total_loss"].append(total_loss.detach().cpu().item())

    def compute(self):
        return {key: avg(values) for key, values in self.metrics.items() if values}

    def log_iter(self, epoch, num_epoch, data_loader):
        batch_size = getattr(data_loader, "batch_size", "unknown")
        total_samples = len(data_loader.dataset) if hasattr(data_loader, "dataset") else "unknown"
        loop = tqdm(
            enumerate(data_loader, 1),
            total=len(data_loader),
            desc=f"Epoch [{epoch}/{num_epoch}] (BS={batch_size}, total={total_samples})",
        )

        self.reset_metrics()
        data_load_time = []
        train_process_time = []

        end_time = time.time()
        for idx, (images, targets) in loop:
            start_time = time.time()
            data_load_time.append(start_time - end_time)

            yield images, targets

            end_time = time.time()
            train_process_time.append(end_time - start_time)

            loop.set_postfix(
                loss=f"{avg(self.metrics['total_loss']):.2f}",
                data=f"{avg(data_load_time):.3f}s/it",
                train=f"{avg(train_process_time):.3f}s/it",
            )
