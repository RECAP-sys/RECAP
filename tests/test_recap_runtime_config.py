from __future__ import annotations

from config.runtime import load_runtime_config


def test_default_config_is_recap_only() -> None:
    config = load_runtime_config("./config/config.yaml")

    assert config.server.continual_learning.training_frame_count == 128
    assert not hasattr(config, "baseline")
    assert not hasattr(config.server, "baselines")


def test_training_frame_count_is_loaded_from_recap_section(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "server:\n  continual_learning:\n    training_frame_count: 32\n",
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.server.continual_learning.training_frame_count == 32
