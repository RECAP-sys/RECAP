from __future__ import annotations

import os


def fs_path(path: str | os.PathLike[str]) -> str:
    text = os.fspath(path)
    if os.name != "nt":
        return text
    if text.startswith("\\\\?\\"):
        return text
    absolute = os.path.abspath(text)
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute.lstrip("\\")
    return "\\\\?\\" + absolute
