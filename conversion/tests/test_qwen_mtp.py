from __future__ import annotations

import pytest

import gguf
from conversion.qwen import _QwenMtpMixin


class _DummyParent:
    def __init__(self, *, hparams: dict, no_mtp: bool = False, model_arch: gguf.MODEL_ARCH, **kwargs):
        self.hparams = hparams
        self.no_mtp = no_mtp
        self.model_arch = model_arch


class _MtpModel(_QwenMtpMixin, _DummyParent):
    pass


@pytest.fixture(autouse=True)
def _reset_mtp_class_state():
    """The mixin uses class attributes that persist across conversions; isolate tests."""
    _MtpModel._original_block_count = None
    _MtpModel.opt_num_mtp_layers = 0
    yield
    _MtpModel._original_block_count = None
    _MtpModel.opt_num_mtp_layers = 0


def test_mtp_count_read_from_text_config():
    model = _MtpModel(
        hparams={
            "num_hidden_layers": 24,
            "text_config": {"num_hidden_layers": 24, "mtp_num_hidden_layers": 1},
        },
        model_arch=gguf.MODEL_ARCH.QWEN3NEXT,
    )
    assert model.block_count == 25


def test_mtp_count_read_from_top_level_config():
    model = _MtpModel(
        hparams={"num_hidden_layers": 24, "mtp_num_hidden_layers": 2},
        model_arch=gguf.MODEL_ARCH.QWEN3NEXT,
    )
    assert model.block_count == 26


def test_mtp_count_falls_back_to_recovered_class_attribute():
    _MtpModel.opt_num_mtp_layers = 1
    model = _MtpModel(
        hparams={"num_hidden_layers": 24},
        model_arch=gguf.MODEL_ARCH.QWEN3NEXT,
    )
    assert model.block_count == 25


def test_missing_mtp_count_raises_actionable_error():
    with pytest.raises(ValueError, match="MTP layer count not found"):
        _MtpModel(
            hparams={"num_hidden_layers": 24},
            model_arch=gguf.MODEL_ARCH.QWEN3NEXT,
        )


def test_no_mtp_skips_mtp_count_lookup():
    model = _MtpModel(
        hparams={"num_hidden_layers": 24},
        no_mtp=True,
        model_arch=gguf.MODEL_ARCH.QWEN3NEXT,
    )
    assert model.block_count == 24
