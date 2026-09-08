from __future__ import annotations

import os
import re
import sys
from collections.abc import MutableSequence, Sequence

_CUDA_DEVICE_RE = re.compile(r"^cuda(?::(?P<index>\d+))?$", re.IGNORECASE)


def configure_default_cuda_visible_devices(
    argv: MutableSequence[str] | None = None,
    *,
    default: str = "0",
    device_flags: Sequence[str] = ("--device",),
) -> None:
    """Set a safe default CUDA visibility before importing torch.

    The caller must run this before any torch import. Existing
    CUDA_VISIBLE_DEVICES values are treated as explicit launch configuration
    and left untouched.
    """

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return

    args = sys.argv if argv is None else argv
    visible_devices = str(default)
    device_ref = _find_option_value(args, tuple(device_flags))
    if device_ref is not None:
        option_index, value_index, option_prefix, value = device_ref
        match = _CUDA_DEVICE_RE.match(str(value).strip())
        if match is not None:
            requested_index = match.group("index")
            if requested_index is not None:
                visible_devices = str(int(requested_index))
                _replace_option_value(
                    args,
                    option_index,
                    value_index,
                    option_prefix,
                    "cuda:0",
                )

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


def _find_option_value(
    argv: MutableSequence[str],
    flags: tuple[str, ...],
) -> tuple[int, int, str | None, str] | None:
    for index, token in enumerate(list(argv)[1:], start=1):
        for flag in flags:
            if token == flag:
                value_index = index + 1
                if value_index < len(argv):
                    return index, value_index, None, str(argv[value_index])
            prefix = f"{flag}="
            if token.startswith(prefix):
                return index, index, prefix, token[len(prefix) :]
    return None


def _replace_option_value(
    argv: MutableSequence[str],
    option_index: int,
    value_index: int,
    option_prefix: str | None,
    value: str,
) -> None:
    if option_prefix is not None:
        argv[option_index] = f"{option_prefix}{value}"
    else:
        argv[value_index] = value
