from pathlib import Path

import pytest

from contrabin.config import ContraBinConfig, CurriculumConfig, ModelConfig


def test_default_config_is_valid():
    cfg = ContraBinConfig()
    assert cfg.model.encoder_name.startswith("microsoft/")
    assert cfg.training.curriculum.total_epochs() == 20
    assert cfg.training.curriculum.stage_for_epoch(0) == "naive"
    assert cfg.training.curriculum.stage_for_epoch(10) == "linear"
    assert cfg.training.curriculum.stage_for_epoch(18) == "nonlinear"


def test_model_config_rejects_bad_dropout():
    with pytest.raises(ValueError):
        ModelConfig(dropout=1.5)


def test_curriculum_total_epochs():
    c = CurriculumConfig(primary_epochs=2, linear_epochs=3, nonlinear_epochs=4)
    assert c.total_epochs() == 9
    assert c.stage_for_epoch(1) == "naive"
    assert c.stage_for_epoch(2) == "linear"
    assert c.stage_for_epoch(5) == "nonlinear"


def test_roundtrip_yaml(tmp_path: Path):
    cfg = ContraBinConfig()
    p = tmp_path / "c.yaml"
    cfg.save_yaml(p)
    cfg2 = ContraBinConfig.from_yaml(p)
    assert cfg2.to_dict() == cfg.to_dict()
