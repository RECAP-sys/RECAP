from __future__ import annotations

import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

_TORCHLENS_FORWARD_LOCK = threading.RLock()
_UNSTABLE_TUPLE_ITERATOR_WARNING = (
    "TorchLens intervention-ready output traversal does not support tuple_iterator; "
    "falling back to BFS without stable output paths."
)


@contextmanager
def torchlens_forward_guard() -> Iterator[None]:
    """Serialize TorchLens tracing with background model forwards.

    TorchLens keeps active logging state at process scope, so unrelated model
    forwards from other threads can corrupt an in-flight trace. Code paths that
    call TorchLens prepare/logging and background teacher inference share this
    guard.
    """

    with _TORCHLENS_FORWARD_LOCK, warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_UNSTABLE_TUPLE_ITERATOR_WARNING,
            category=UserWarning,
        )
        yield
