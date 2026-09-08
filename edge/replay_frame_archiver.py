from __future__ import annotations

import queue
import threading
import zipfile
from pathlib import Path

import cv2
from loguru import logger


class ReplayFrameArchiver:
    def __init__(
        self,
        run_dir: Path,
        *,
        enabled: bool,
        jpeg_quality: int = 90,
        queue_size: int = 64,
        archive_chunk_max_bytes: int = 67108864,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.enabled = bool(enabled)
        self.jpeg_quality = min(100, max(1, int(jpeg_quality)))
        self.archive_chunk_max_bytes = max(1, int(archive_chunk_max_bytes))
        self.frame_dir = self.run_dir / "replay_frames"
        capacity = max(1, int(queue_size))
        self._queue: queue.Queue[tuple[int, object] | None] = queue.Queue(
            maxsize=capacity
        )
        self._slots = threading.BoundedSemaphore(capacity)
        self._thread: threading.Thread | None = None
        self._failures: dict[int, str] = {}
        if self.enabled:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(
                target=self._worker,
                name="teacher-replay-frame-writer",
                daemon=True,
            )
            self._thread.start()

    @staticmethod
    def relative_frame_path(frame_index: int) -> str:
        return f"replay_frames/frame_{int(frame_index):09d}.jpg"

    def enqueue(self, frame_index: int, frame) -> str | None:
        if not self.enabled:
            return None
        resolved_frame_index = int(frame_index)
        if not self._slots.acquire(blocking=False):
            self._failures[resolved_frame_index] = "snapshot queue full"
            return None
        try:
            copied_frame = frame.copy()
        except Exception:
            self._slots.release()
            raise
        try:
            self._queue.put_nowait((resolved_frame_index, copied_frame))
        except queue.Full:
            self._slots.release()
            self._failures[resolved_frame_index] = "snapshot queue full"
            return None
        return self.relative_frame_path(frame_index)

    def close(self) -> list[Path]:
        if not self.enabled:
            return []
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join()
        archives = self._build_archives()
        if self._failures:
            logger.warning(
                "Teacher replay snapshot failures: count={}",
                len(self._failures),
            )
        return archives

    @property
    def failures(self) -> dict[int, str]:
        return dict(self._failures)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                frame_index, frame = item
                path = self.run_dir / self.relative_frame_path(frame_index)
                ok = cv2.imwrite(
                    str(path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not ok:
                    self._failures[frame_index] = "cv2.imwrite returned false"
            except Exception as exc:
                if item is not None:
                    self._failures[item[0]] = str(exc)
            finally:
                if item is not None:
                    self._slots.release()
                self._queue.task_done()

    def _build_archives(self) -> list[Path]:
        files = sorted(self.frame_dir.glob("frame_*.jpg"))
        if not files:
            return []
        archives: list[Path] = []
        chunk_index = 0
        chunk_size = 22
        archive: zipfile.ZipFile | None = None
        try:
            for path in files:
                frame_index = int(path.stem.split("_")[-1])
                arcname = self.relative_frame_path(frame_index)
                entry_size = path.stat().st_size + 76 + (2 * len(arcname.encode("utf-8")))
                if entry_size + 22 > self.archive_chunk_max_bytes:
                    self._failures[frame_index] = (
                        "encoded JPEG exceeds archive_chunk_max_bytes"
                    )
                    continue
                if archive is None or (
                    chunk_size > 22
                    and chunk_size + entry_size > self.archive_chunk_max_bytes
                ):
                    if archive is not None:
                        archive.close()
                    chunk_index += 1
                    archive_path = self.run_dir / f"replay_frames_{chunk_index:04d}.zip"
                    archives.append(archive_path)
                    archive = zipfile.ZipFile(
                        archive_path,
                        mode="w",
                        compression=zipfile.ZIP_STORED,
                    )
                    chunk_size = 22
                archive.write(path, arcname=arcname)
                chunk_size += entry_size
        finally:
            if archive is not None:
                archive.close()
        for archive_path in archives:
            if archive_path.stat().st_size > self.archive_chunk_max_bytes:
                raise RuntimeError(
                    f"replay archive exceeds configured chunk size: {archive_path}"
                )
        return archives
