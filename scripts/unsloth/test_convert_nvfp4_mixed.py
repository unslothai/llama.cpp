#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for the compressed-tensors mixed-precision NVFP4 path in conversion/base.py.
Run: python3 scripts/unsloth/test_convert_nvfp4_mixed.py

Recognising a mixed-precision checkpoint means the NVFP4 branch of dequant_model now
runs on real inputs instead of being a bare `pass`. Everything the NVFP4 group does not
consume lands in that branch, and the only dequantizer it has is dequant_simple, which
is correct for exactly one shape of input: an unpacked weight with one scale per row.

So the thing worth testing is not that a supported checkpoint converts. It is that an
UNSUPPORTED residual group is refused rather than quietly multiplied by a scale and
written out, because a converter's output is a file someone keeps and uses for months
and a wrong number in it never announces itself.

Two ways the branch could be handed something dequant_simple cannot dequantize:

  1. A "pack-quantized" residual group. Its weights are nibble-packed ints needing
     dequant_packed. It passes a strategy-only guard because its strategy really is
     "channel". And prepare_tensors renames every `.weight_packed` to `.weight` BEFORE
     dequant_model runs, so by the time the branch sees it, the `weight_name not in
     model_tensors` guard is testing a name that now exists. The guard looks correct and
     is inert. That is `test_pack_quantized_residual_is_refused`, and the config it uses
     is copied verbatim from unsloth/gemma-4-E2B-it-NVFP4 and -E4B-it-NVFP4, which ship
     exactly this group_2 over embed_tokens_per_layer.

  2. An NVFP4 tensor whose block geometry _generate_nvfp4_tensors skipped. It stays uint8
     and packed. For most widths the shapes then fail to broadcast and you get a loud
     error, but when the scale is [out, 1] it broadcasts cleanly and dequant_simple
     happily returns packed nibbles scaled as if they were weights. That is
     `test_surviving_packed_uint8_is_refused`.

dequant_model only reads self._is_nvfp4, self.model_tensors, self.hparams,
self._fp8_as_q8 and self._fp8_dequantized, so these drive it through a shim rather than
running a full conversion. No tokenizer, no network, no model download.
"""
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}{': ' + detail if detail else ''}")
        FAILS.append(name)


try:
    import torch
    from conversion.base import ModelBase
except ImportError as e:  # pragma: no cover
    print(f"SKIP: {e}. Install requirements/requirements-convert_hf_to_gguf.txt to run this.")
    sys.exit(0)


class Shim:
    """The four attributes dequant_model actually reads."""

    def __init__(self, tensors, quant_config, is_nvfp4=True, fp8_as_q8=False):
        self.model_tensors = {k: (lambda v=v: v) for k, v in tensors.items()}
        self.hparams = {"quantization_config": quant_config}
        self._is_nvfp4 = is_nvfp4
        self._fp8_as_q8 = fp8_as_q8
        self._fp8_dequantized = set()

    def run(self):
        ModelBase.dequant_model(self)

    def get(self, name):
        return self.model_tensors[name]()


def nvfp4_group(targets):
    return {
        "format": "nvfp4-pack-quantized",
        "targets": targets,
        "weights": {
            "num_bits": 4, "type": "float", "strategy": "tensor_group",
            "group_size": 16, "block_structure": None, "symmetric": True,
            "scale_dtype": "torch.float8_e4m3fn",
        },
    }


def fp8_channel_group(targets):
    return {
        "format": "float-quantized",
        "targets": targets,
        "weights": {
            "num_bits": 8, "type": "float", "strategy": "channel",
            "group_size": None, "block_structure": None, "symmetric": True,
        },
    }


# Copied verbatim from unsloth/gemma-4-E2B-it-NVFP4 and unsloth/gemma-4-E4B-it-NVFP4
# config.json, quantization_config.config_groups.group_2. Both ship this identically.
GEMMA4_PACK_QUANTIZED_GROUP = {
    "format": "pack-quantized",
    "input_activations": None,
    "output_activations": None,
    "targets": ["re:.*embed_tokens_per_layer$"],
    "weights": {
        "actorder": None, "block_structure": None, "dynamic": False,
        "group_size": None, "num_bits": 8, "observer": "memoryless_minmax",
        "observer_kwargs": {}, "scale_dtype": None, "strategy": "channel",
        "symmetric": True, "type": "int", "zp_dtype": None,
    },
}


def residual_tensors(prefix="model.layers.0.self_attn.q_proj"):
    """The state dequant_model actually sees. _generate_nvfp4_tensors has already run and
    consumed every NVFP4 weight/scale pair, so what is left is the residual group only."""
    return {
        f"{prefix}.weight": torch.zeros(4, 8, dtype=torch.float8_e4m3fn),
        f"{prefix}.weight_scale": torch.ones(4, 1),
    }


def raises(shim):
    try:
        shim.run()
    except BaseException as e:  # noqa: BLE001 - the type is part of what we assert
        return e
    return None


# --------------------------------------------------------------------------------------
# 1. The defect: a pack-quantized residual group must be refused, not dequant_simple'd.
#    Before the fix this returned cleanly and left embed_tokens_per_layer.weight as a
#    dequant_simple lambda over nibble-packed int32. Exit code 0, wrong file.
# --------------------------------------------------------------------------------------
def test_pack_quantized_residual_is_refused():
    tensors = residual_tensors()
    tensors.update({
        # prepare_tensors has already renamed .weight_packed -> .weight, which is exactly
        # why the `weight_name not in model_tensors` guard does not catch this.
        "model.embed_tokens_per_layer.weight": torch.zeros(8, 16, dtype=torch.int32),
        "model.embed_tokens_per_layer.weight_scale": torch.ones(8, 1, dtype=torch.bfloat16),
        "model.embed_tokens_per_layer.weight_shape": torch.tensor([8, 64]),
    })
    shim = Shim(tensors, {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": fp8_channel_group(["re:.*self_attn\\.(q|k|v|o)_proj$"]),
            "group_1": nvfp4_group(["re:.*mlp\\.(gate|up|down)_proj$"]),
            "group_2": GEMMA4_PACK_QUANTIZED_GROUP,
        },
    })
    err = raises(shim)
    check("pack-quantized residual raises", isinstance(err, NotImplementedError),
          f"got {err!r}")
    check("pack-quantized message names the format",
          err is not None and "pack-quantized" in str(err), f"got {err!r}")
    check("pack-quantized weight was not silently dequantized",
          err is not None,
          "dequant_model returned cleanly, embed_tokens_per_layer is now wrong numbers")


# --------------------------------------------------------------------------------------
# 2. Control: the shape this PR exists to support must still convert. A guard that
#    refuses everything would pass test 1 and be useless.
# --------------------------------------------------------------------------------------
def test_nvfp4_plus_fp8_channel_still_works():
    w = (torch.arange(32, dtype=torch.float32).reshape(4, 8) / 8.0).to(torch.float8_e4m3fn)
    s = torch.tensor([[2.0], [3.0], [4.0], [5.0]])
    tensors = residual_tensors()
    tensors.update({
        "model.layers.0.self_attn.q_proj.weight": w,
        "model.layers.0.self_attn.q_proj.weight_scale": s,
        "model.layers.0.self_attn.q_proj.input_scale": torch.tensor(1.0),
        "model.layers.0.self_attn.k_scale": torch.tensor(1.0),
        "model.layers.0.self_attn.v_scale": torch.tensor(1.0),
    })
    shim = Shim(tensors, {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": fp8_channel_group(["re:.*self_attn\\.(q|k|v|o)_proj$"]),
            "group_1": nvfp4_group(["re:.*mlp\\.(gate|up|down)_proj$"]),
        },
    }, fp8_as_q8=True)
    err = raises(shim)
    check("nvfp4 + fp8-channel does not raise", err is None, f"got {err!r}")
    if err is not None:
        return
    got = shim.get("model.layers.0.self_attn.q_proj.weight")
    check("fp8 residual dequantized to the right numbers",
          torch.allclose(got, w.float() * s), f"got {got}")
    for sidecar in ("weight_scale", "input_scale"):
        check(f"{sidecar} sidecar dropped",
              f"model.layers.0.self_attn.q_proj.{sidecar}" not in shim.model_tensors)
    for sidecar in ("k_scale", "v_scale"):
        check(f"{sidecar} sidecar dropped",
              f"model.layers.0.self_attn.{sidecar}" not in shim.model_tensors)
    check("fp8 weight recorded for --fp8-as-q8",
          "model.layers.0.self_attn.q_proj.weight" in shim._fp8_dequantized)


# --------------------------------------------------------------------------------------
# 3. A residual group with a block_structure is refused, and says so. dequant_simple
#    would apply a 2-D grid of scales as if it were per-row.
# --------------------------------------------------------------------------------------
def test_block_structure_residual_is_refused():
    group = fp8_channel_group(["re:.*self_attn\\.(q|k|v|o)_proj$"])
    group["weights"]["block_structure"] = [128, 128]
    shim = Shim(residual_tensors(), {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {"group_0": group, "group_1": nvfp4_group(["re:.*mlp.*$"])},
    })
    err = raises(shim)
    check("block_structure residual raises", isinstance(err, NotImplementedError),
          f"got {err!r}")
    # The message used to say "a 'channel' group is not supported", naming the field
    # that was fine and hiding the field that was not.
    check("block_structure message names block_structure",
          err is not None and "block_structure" in str(err), f"got {err!r}")


def test_non_channel_strategy_residual_is_refused():
    for strategy in ("tensor", "group", "token", None):
        group = fp8_channel_group(["re:.*self_attn.*$"])
        group["weights"]["strategy"] = strategy
        shim = Shim(residual_tensors(), {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": {"group_0": group, "group_1": nvfp4_group(["re:.*mlp.*$"])},
        })
        err = raises(shim)
        check(f"strategy={strategy!r} residual raises",
              isinstance(err, NotImplementedError), f"got {err!r}")


# --------------------------------------------------------------------------------------
# 4. Malformed and older exports must fail as NotImplementedError, never as a KeyError
#    or AttributeError from an unguarded dict access.
# --------------------------------------------------------------------------------------
def test_missing_weights_key_is_not_a_keyerror():
    for weights in (None, {}):
        group = {"format": "float-quantized", "targets": ["re:.*self_attn.*$"], "weights": weights}
        shim = Shim(residual_tensors(), {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": {"group_0": group, "group_1": nvfp4_group(["re:.*mlp.*$"])},
        })
        err = raises(shim)
        check(f"weights={weights!r} raises NotImplementedError",
              isinstance(err, NotImplementedError), f"got {type(err).__name__}: {err}")

    # weights absent entirely. `weight_config = tuple(groups.values())[0]["weights"]` used
    # to run unconditionally just after the gate and KeyError here, even though the NVFP4
    # branch never uses weight_config.
    group = {"format": "float-quantized", "targets": ["re:.*self_attn.*$"]}
    shim = Shim(residual_tensors(), {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {"group_0": group, "group_1": nvfp4_group(["re:.*mlp.*$"])},
    })
    err = raises(shim)
    check("absent weights key raises NotImplementedError not KeyError",
          isinstance(err, NotImplementedError), f"got {type(err).__name__}: {err}")


# --------------------------------------------------------------------------------------
# 5. An NVFP4 tensor _generate_nvfp4_tensors skipped stays packed uint8. With a [out, 1]
#    scale it broadcasts cleanly, so this is the one shape where the old code produced a
#    file with no error at all.
# --------------------------------------------------------------------------------------
def test_surviving_packed_uint8_is_refused():
    shim = Shim({
        "model.layers.0.mlp.down_proj.weight": torch.full((4, 8), 0x42, dtype=torch.uint8),
        "model.layers.0.mlp.down_proj.weight_scale": torch.ones(4, 1).to(torch.float8_e4m3fn),
    }, {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": fp8_channel_group(["re:.*self_attn.*$"]),
            "group_1": nvfp4_group(["re:.*mlp.*$"]),
        },
    })
    err = raises(shim)
    check("surviving packed uint8 raises", isinstance(err, NotImplementedError),
          f"got {err!r}")
    check("packed uint8 message names the dtype",
          err is not None and "uint8" in str(err), f"got {err!r}")


# --------------------------------------------------------------------------------------
# 6. The gate itself. any() must not pull a mixed-precision checkpoint with no NVFP4
#    group into the NVFP4 path.
# --------------------------------------------------------------------------------------
def test_mixed_precision_without_nvfp4_still_refused():
    shim = Shim({
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 8, dtype=torch.float8_e4m3fn),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.ones(4, 1),
    }, {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": fp8_channel_group(["re:.*self_attn.*$"]),
            "group_1": fp8_channel_group(["re:.*mlp.*$"]),
        },
    }, is_nvfp4=False)
    err = raises(shim)
    check("mixed-precision with no NVFP4 group is refused",
          isinstance(err, NotImplementedError) and "multiple config groups" in str(err),
          f"got {err!r}")


# --------------------------------------------------------------------------------------
# 7. Single-group formats must be untouched by all of the above.
# --------------------------------------------------------------------------------------
def test_single_group_float_quantized_unaffected():
    w = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
    s = torch.full((4, 1), 2.0)
    shim = Shim({
        "model.layers.0.self_attn.q_proj.weight": w,
        "model.layers.0.self_attn.q_proj.weight_scale": s,
    }, {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {"group_0": fp8_channel_group(["re:.*self_attn.*$"])},
    }, is_nvfp4=False, fp8_as_q8=True)
    err = raises(shim)
    check("single-group float-quantized does not raise", err is None, f"got {err!r}")
    if err is None:
        check("single-group float-quantized dequantizes correctly",
              torch.allclose(shim.get("model.layers.0.self_attn.q_proj.weight"), w.float() * s))


if __name__ == "__main__":
    for fn in (
        test_pack_quantized_residual_is_refused,
        test_nvfp4_plus_fp8_channel_still_works,
        test_block_structure_residual_is_refused,
        test_non_channel_strategy_residual_is_refused,
        test_missing_weights_key_is_not_a_keyerror,
        test_surviving_packed_uint8_is_refused,
        test_mixed_precision_without_nvfp4_still_refused,
        test_single_group_float_quantized_unaffected,
    ):
        print(f"-- {fn.__name__}")
        try:
            fn()
        except Exception:
            traceback.print_exc()
            FAILS.append(fn.__name__)

    print()
    if FAILS:
        print(f"{len(FAILS)} failure(s): {', '.join(FAILS)}")
        sys.exit(1)
    print("all passed")
