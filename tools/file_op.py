import os
import shutil

from loguru import logger


def creat_folder(folder_path):
    frames_path = os.path.join(folder_path, "frames")
    logger.debug("creat {}".format(frames_path))
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)


def clear_folder(folder_path, preserve=None):
    logger.debug("clear floder")

    if not os.path.exists(folder_path):
        return
    preserve = set(preserve or ())
    for filename in os.listdir(folder_path):
        if filename in preserve:
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            elif os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.debug(f"Failed to delete path: {file_path}. Reason: {e}")


def sample_files(root, indexs):
    logger.debug("clear index {}".format(indexs))
    for filename in os.listdir(root):
        if int(filename.split(".")[0]) not in indexs:
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"remove error: {e}")
