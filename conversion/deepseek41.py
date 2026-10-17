from __future__ import annotations

import os
import re
from pathlib import Path

from typing import Any, Callable, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf, logger
from .deepseek import DeepseekV4Model


# ---------------------------------------------------------------------------------------------------------------------
# Engram hash constants, ported from inference/engram.py of deepseek-ai/DeepSeek-V4.1-Flash-1017.
# They are written to GGUF so that the runtime never recomputes them (the compressed token map needs the HF
# tokenizers normalizers, the multipliers numpy's PCG64 stream).
# ---------------------------------------------------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    # deterministic Miller-Rabin for n < 3.3e24
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def engram_primes(n_layers: int, max_ngram_size: int, n_heads: int, vocab_size: int) -> list[list[list[int]]]:
    """EngramLayout.from_args: [layer][ngram order 2..max][head] bucket moduli, never reused across layers."""
    seen: set[int] = set()
    out: list[list[list[int]]] = []
    for _ in range(n_layers):
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, cur = [], vocab_size - 1
            for _ in range(n_heads):
                cand = cur + 1
                while not _is_prime(cand) or cand in seen:
                    cand += 1
                cur = cand
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(sizes)
        out.append(per_ngram)
    return out


def engram_multipliers(layer_ids: list[int], max_ngram_size: int, compressed_vocab_size: int) -> list[list[int]]:
    """compute_hash_multipliers: odd, bounded so that token_id * multiplier cannot overflow int64."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append([int(v) * 2 + 1 for v in values])
    return rows


def engram_compressed_token_map(tokenizer_dir: Path) -> tuple[list[int], int]:
    """build_compressed_token_map: token id -> compressed id (tokens that normalize alike collapse)."""
    from tokenizers import Regex, normalizers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


# ---------------------------------------------------------------------------------------------------------------------
# Streaming raw tensors: gguf_writer keeps every added tensor until the final write and only needs
# shape / dtype / nbytes / tofile, so producing the bytes chunk by chunk at write time keeps memory flat
# (an Engram table is ~101 GB, a stacked MXFP4 expert tensor ~2.4 GB).
# ---------------------------------------------------------------------------------------------------------------------

class _StreamedBytes:
    def __init__(self, shape: tuple[int, ...], chunks: Callable[[], Iterable[np.ndarray]]):
        self.shape = shape
        self.dtype = np.dtype(np.uint8)
        self.nbytes = int(np.prod(shape))
        self._chunks = chunks

    def tofile(self, fout) -> None:
        written = 0
        for chunk in self._chunks():
            chunk = np.ascontiguousarray(chunk, dtype=np.uint8)
            chunk.tofile(fout)
            written += chunk.nbytes
        if written != self.nbytes:
            raise ValueError(f"streamed {written} bytes, expected {self.nbytes}")

    def byteswap(self, inplace: bool = False):
        return self  # raw bytes


def _e8m0_check(scale_u8: np.ndarray, name: str, lo: int, hi: int) -> None:
    if scale_u8.size == 0:
        return
    smin, smax = int(scale_u8.min()), int(scale_u8.max())
    if smin < lo or smax > hi:
        raise ValueError(f"{name}: E8M0 scale exponents [{smin}, {smax}] outside the exact range [{lo}, {hi}] "
                         f"(0xFF is NaN, very small exponents underflow BF16)")


@ModelBase.register("DeepseekV41ForCausalLM")
@ModelBase.example("deepseek-ai/DeepSeek-V4.1-Flash-1017")
class DeepseekV41Model(DeepseekV4Model):
    """DeepSeek-V4.1 text model (arch deepseek41), see TENSORS.md in the port branch.

    Lossless reference layout: FP8 32x32-block linears dequantised to BF16 (exact: e4m3 * 2^k fits BF16), routed
    experts copied bit-exact as MXFP4, Engram tables copied bit-exact as MXFP8. DSpark blocks (mtp.0..2) go to a
    separate GGUF with --mtp. The vision tower is not converted yet.

    Environment overrides:
      DSV41_DENSE_TYPE   bf16 (default, lossless) | q8_0  for the dequantised FP8 linears
      DSV41_ENGRAM_TYPE  mxfp8 (default, lossless) | bf16 (lossless, 2x larger) | f32
    """
    model_arch = gguf.MODEL_ARCH.DEEPSEEK41
    supports_mtp_export = True

    # tensors that stay in their source float dtype (bf16 in the checkpoint, never quantised)
    _ENGRAM_EMBED_RE = re.compile(r"layers\.(\d+)\.engram\.embed\.(weight|scale)$")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams
        self._dense_type = os.environ.get("DSV41_DENSE_TYPE", "bf16").lower()
        self._engram_type = os.environ.get("DSV41_ENGRAM_TYPE", "mxfp8").lower()
        if self._dense_type not in ("bf16", "q8_0"):
            raise ValueError(f"DSV41_DENSE_TYPE={self._dense_type!r}")
        if self._engram_type not in ("mxfp8", "bf16", "f32"):
            raise ValueError(f"DSV41_ENGRAM_TYPE={self._engram_type!r}")

        qc = hp.get("quantization_config") or {}
        self._fp8_block = tuple(qc.get("weight_block_size") or (128, 128))
        if qc.get("scale_fmt", "ue8m0") != "ue8m0":
            raise ValueError(f"unsupported FP8 scale format {qc.get('scale_fmt')!r}")

        # the V4 constructor installed the V4 template; V4.1 has its own
        template_path = Path(__file__).parent.parent / "models" / "templates" / "deepseek-ai-DeepSeek-V4.1.jinja"
        if template_path.is_file():
            self.gguf_writer.add_chat_template(template_path.read_text(encoding="utf-8"))

    # ---- tensor selection -------------------------------------------------------------------------------------------

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        hp = {**self.hparams, **self.hparams.get("text_config", {})}  # text_config is merged only after indexing
        type(self)._dsv4_main_layers = hp["num_hidden_layers"]
        type(self)._dsv4_nextn_layers = hp.get("num_nextn_predict_layers", 0)
        return TextModel.index_tensors(self, remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.startswith(("vision.", "aligner.", "image_")):
            return None  # vision tower: mmproj not converted yet
        if name.startswith("mtp."):
            if not cls.mtp_only:
                cls._skipped_mtp_tensors += 1
                return None
            m = re.match(r"mtp\.(\d+)\.(.+)$", name)
            if m is None:
                raise ValueError(f"Unexpected DeepSeek-V4.1 DSpark tensor {name!r}")
            stage, rest = int(m.group(1)), m.group(2)
            assert cls._dsv4_main_layers is not None
            if stage >= cls._dsv4_nextn_layers:
                raise ValueError(f"Unexpected DSpark stage {stage}")
            bid = cls._dsv4_main_layers + stage
            if rest.startswith(("markov_head.", "confidence_head.")):
                return f"dspark.{rest}", gen
            return f"layers.{bid}.{rest}", gen
        if cls.mtp_only and name not in ("embed.weight", "norm.weight", "head.weight"):
            return None
        return TextModel.filter_tensors((name, gen))

    # ---- hparams ----------------------------------------------------------------------------------------------------

    def set_gguf_parameters(self):
        TextModel.set_gguf_parameters(self)
        hp = self.hparams
        w = self.gguf_writer
        n_main = hp["num_hidden_layers"]

        w.add_rope_dimension_count(hp["qk_rope_head_dim"])
        w.add_q_lora_rank(hp["q_lora_rank"])
        w.add_sliding_window(hp["sliding_window"])
        w.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        w.add_expert_shared_count(hp["n_shared_experts"])
        w.add_expert_weights_scale(hp["routed_scaling_factor"])
        w.add_expert_weights_norm(hp["norm_topk_prob"])
        w.add_swiglu_clamp_exp([hp["swiglu_limit"]] * self.block_count)
        w.add_swiglu_clamp_shexp([hp["swiglu_limit"]] * self.block_count)

        w.add_indexer_head_count(hp["index_n_heads"])
        w.add_indexer_key_length(hp["index_head_dim"])
        w.add_indexer_top_k(hp["index_topk"])

        w.add_attention_output_group_count(hp["o_groups"])
        w.add_attention_output_lora_rank(hp["o_lora_rank"])
        w.add_attention_compress_ratios(hp["compress_ratios"])
        w.add_attention_compress_rope_freq_base(hp["compress_rope_theta"])
        w.add_hyper_connection_count(hp["hc_mult"])
        w.add_hyper_connection_sinkhorn_iterations(hp["hc_sinkhorn_iters"])
        w.add_hyper_connection_epsilon(hp["hc_eps"])
        w.add_embedding_length_out(hp["hidden_size"] * hp["hc_mult"])

        # cross-layer sharing and candidate blocks
        w.add_attention_kv_source_layers(hp["kv_source_layer_ids"])
        w.add_indexer_source_layers(hp["index_source_layer_ids"])
        w.add_indexer_candidate_source_layer(int(hp.get("candidate_source_layer_id", -1)))
        w.add_indexer_candidate_top_k_blocks(hp["candidate_topk_blocks"])
        w.add_indexer_candidate_block_size(hp["candidate_block_size"])

        # Engram
        layer_ids = list(hp["engram_layer_ids"])
        max_ngram, n_heads = hp["engram_max_ngram_size"], hp["engram_n_heads"]
        primes = engram_primes(len(layer_ids), max_ngram, n_heads, hp["engram_vocab_size"])
        for li, layer_primes in enumerate(primes):
            total = sum(p for per in layer_primes for p in per)
            if total != hp["engram_num_embeddings"][li]:
                raise ValueError(f"engram layer {layer_ids[li]}: prime sum {total} != num_embeddings {hp['engram_num_embeddings'][li]}")
        token_map, cvocab = engram_compressed_token_map(self.dir_model if self.target_model_dir is None else self.target_model_dir)
        if cvocab != hp["engram_compressed_vocab_size"]:
            raise ValueError(f"compressed vocab {cvocab} != engram_compressed_vocab_size {hp['engram_compressed_vocab_size']}")
        if len(token_map) != hp["vocab_size"]:
            raise ValueError(f"token map has {len(token_map)} entries, vocab_size is {hp['vocab_size']}")
        mults = engram_multipliers(layer_ids, max_ngram, cvocab)
        w.add_engram_layers(layer_ids)
        w.add_engram_num_embeddings(hp["engram_num_embeddings"])
        w.add_engram_max_ngram_size(max_ngram)
        w.add_engram_vocab_size(hp["engram_vocab_size"])
        w.add_engram_head_count(n_heads)
        w.add_engram_head_dim(hp["engram_head_dim"])
        w.add_engram_pad_token_id(hp["engram_pad_token_id"])
        w.add_engram_compressed_vocab_size(cvocab)
        w.add_engram_token_map(token_map)
        w.add_engram_multipliers([m for row in mults for m in row])
        w.add_engram_primes([p for layer in primes for per in layer for p in per])
        logger.info(f"engram: compressed vocab {cvocab}, pad -> {token_map[hp['engram_pad_token_id']]}, multipliers {mults}")

        # DSpark (block layout only matters to the --mtp file, but the keys are cheap)
        if "dspark_block_size" in hp:
            w.add_dspark_block_size(hp["dspark_block_size"])
            w.add_dspark_noise_token_id(hp["dspark_noise_token_id"])
            w.add_dspark_target_layers(hp["dspark_target_layer_ids"])
            w.add_dspark_markov_rank(hp["dspark_markov_rank"])
            w.add_dspark_expert_count(hp["dspark_n_routed_experts"])
            w.add_dspark_expert_used_count(hp["dspark_num_experts_per_tok"])
        if self.mtp_only:
            w.add_nextn_predict_layers(hp.get("num_nextn_predict_layers", 0))
        assert self.block_count == n_main + (hp.get("num_nextn_predict_layers", 0) if self.mtp_only else 0)

    # ---- weights ----------------------------------------------------------------------------------------------------

    def dequant_model(self):
        """FP8 (e4m3, ue8m0 scales) linears -> f32, exact. The block is derived per tensor from the scale shape and
        checked against quantization_config.weight_block_size (V4 used 128, V4.1 uses 32x32)."""
        fp8_dtypes = self._float8_dtypes()
        bo_cfg, bi_cfg = self._fp8_block
        to_remove: list[str] = []

        def dequant(weight: Tensor, scale: Tensor, bo: int, bi: int) -> Tensor:
            out_f, in_f = weight.shape
            s = self._e8m0_to_float(scale)
            s = s.repeat_interleave(bo, 0)[:out_f].repeat_interleave(bi, 1)[:, :in_f]
            return weight.float() * s

        for name in list(self.model_tensors.keys()):
            if not name.endswith(".scale") or self._ENGRAM_EMBED_RE.match(name):
                continue
            weight_name = name.removesuffix(".scale") + ".weight"
            if weight_name not in self.model_tensors:
                continue
            wgen, sgen = self.model_tensors[weight_name], self.model_tensors[name]
            w_meta = wgen()
            if w_meta.dtype not in fp8_dtypes:
                continue  # MXFP4 experts (I8 packed) are handled in generate_extra_tensors
            out_f, in_f = w_meta.shape
            s_meta = sgen()
            srows, scols = s_meta.shape
            bo, bi = -(-out_f // srows), -(-in_f // scols)
            if (bo, bi) != (bo_cfg, bi_cfg) or -(-out_f // bo) != srows or -(-in_f // bi) != scols:
                raise ValueError(f"{weight_name}: FP8 block {(bo, bi)} from scale {tuple(s_meta.shape)} "
                                 f"does not match weight_block_size {(bo_cfg, bi_cfg)}")
            # exactness: e4m3 (|x| in [2^-9, 448]) * 2^(e-127) must stay inside BF16's range
            _e8m0_check(LazyTorchTensor.to_eager(s_meta).view(torch.uint8).numpy(), name, 3, 246)
            self.model_tensors[weight_name] = lambda w=wgen, s=sgen, bo=bo, bi=bi: dequant(w(), s(), bo, bi)
            self._dsv4_fp8_dequantized.add(weight_name)
            to_remove.append(name)

        for name in to_remove:
            del self.model_tensors[name]

    def _write_mxfp4_experts(self, bid: int, n_experts: int) -> list[str]:
        consumed: list[str] = []
        for proj, key in (("w1", gguf.MODEL_TENSOR.FFN_GATE_EXP), ("w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP),
                          ("w3", gguf.MODEL_TENSOR.FFN_UP_EXP)):
            loaders = []
            for eid in range(n_experts):
                wn, sn = f"layers.{bid}.ffn.experts.{eid}.{proj}.weight", f"layers.{bid}.ffn.experts.{eid}.{proj}.scale"
                if wn not in self.model_tensors or sn not in self.model_tensors:
                    raise KeyError(f"Missing routed expert tensors {wn}")
                loaders.append((self.model_tensors[wn], self.model_tensors[sn]))
                consumed += [wn, sn]
            extra = sorted(n for n in self.model_tensors
                           if re.match(rf"layers\.{bid}\.ffn\.experts\.(\d+)\.{proj}\.weight$", n)
                           and int(n.split(".")[4]) >= n_experts)
            if extra:
                raise KeyError(f"layer {bid}: unexpected experts beyond {n_experts}: {extra[:3]}")
            data = self._mxfp4_expert_tensor(loaders)
            new_name = self.format_tensor_name(key, bid)
            self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)
        return consumed

    def _engram_embed_tensor(self, bid: int) -> list[str]:
        wn, sn = f"layers.{bid}.engram.embed.weight", f"layers.{bid}.engram.embed.scale"
        wgen, sgen = self.model_tensors[wn], self.model_tensors[sn]
        rows, dim = wgen().shape
        if tuple(sgen().shape) != (rows, dim // 32) or dim % 32 != 0:
            raise ValueError(f"{wn}: scale shape {tuple(sgen().shape)} is not 1x32 blocks of {(rows, dim)}")
        li = list(self.hparams["engram_layer_ids"]).index(bid)
        if rows != self.hparams["engram_num_embeddings"][li]:
            raise ValueError(f"{wn}: {rows} rows != engram_num_embeddings {self.hparams['engram_num_embeddings'][li]}")
        new_name = self.format_tensor_name(gguf.MODEL_TENSOR.ENGRAM_EMBD, bid)
        part = self._tensor_part(wn)
        chunk_rows = 1 << 20

        def raw_chunks() -> Iterable[tuple[np.ndarray, np.ndarray]]:
            from safetensors import safe_open
            with safe_open(part, framework="pt") as f:
                ws, ss = f.get_slice(wn), f.get_slice(sn)
                for r0 in range(0, rows, chunk_rows):
                    r1 = min(rows, r0 + chunk_rows)
                    q = ws[r0:r1].view(torch.uint8).numpy().reshape(-1, 32)
                    e = ss[r0:r1].view(torch.uint8).numpy().reshape(-1, 1)
                    yield q, e

        if self._engram_type == "mxfp8":
            def chunks() -> Iterable[np.ndarray]:
                for q, e in raw_chunks():
                    if e.max() == 0xFF:
                        raise ValueError(f"{wn}: E8M0 NaN scale (0xFF)")
                    if ((q & 0x7F) == 0x7F).any():
                        raise ValueError(f"{wn}: E4M3 NaN code")
                    yield np.concatenate([e, q], axis=1)
            data = _StreamedBytes((rows, dim // 32 * 33), chunks)
            self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP8)  # type: ignore[arg-type]
        else:
            to_bf16 = self._engram_type == "bf16"

            def chunks() -> Iterable[np.ndarray]:
                for q, e in raw_chunks():
                    _e8m0_check(e, wn, 3, 246)
                    v = torch.from_numpy(q).view(torch.float8_e4m3fn).float() * torch.exp2(torch.from_numpy(e).float() - 127)
                    v = v.reshape(-1, dim)
                    yield (v.bfloat16().view(torch.uint8) if to_bf16 else v.view(torch.uint8)).numpy()
            esz = 2 if to_bf16 else 4
            data = _StreamedBytes((rows, dim * esz), chunks)
            qt = gguf.GGMLQuantizationType.BF16 if to_bf16 else gguf.GGMLQuantizationType.F32
            self.gguf_writer.add_tensor(new_name, data, raw_shape=(rows, dim), raw_dtype=qt)  # type: ignore[arg-type]
        logger.info(f"{new_name}: Engram table {rows} x {dim} copied as {self._engram_type.upper()}")
        return [wn, sn]

    def _tensor_part(self, name: str) -> Path:
        import json
        with open(self.dir_model / "model.safetensors.index.json", "r", encoding="utf-8") as f:
            return self.dir_model / json.load(f)["weight_map"][name]

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if self._dsv4_mxfp4_generated:
            return ()
        hp = self.hparams
        consumed: list[str] = []
        n_main = hp["num_hidden_layers"]
        for bid in range(self.block_count):
            if self.mtp_only and bid < n_main:
                continue
            n_exp = hp["n_routed_experts"] if bid < n_main else hp["dspark_n_routed_experts"]
            consumed += self._write_mxfp4_experts(bid, n_exp)
        if not self.mtp_only:
            for bid in hp["engram_layer_ids"]:
                consumed += self._engram_embed_tensor(bid)
        for name in consumed:
            del self.model_tensors[name]
        self._dsv4_mxfp4_generated = True
        return ()

    _LAYER_MAP: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
        "attn.indexer.wk.weight": (gguf.MODEL_TENSOR.INDEXER_ATTN_K, ".weight"),
        "attn.indexer.k_norm.weight": (gguf.MODEL_TENSOR.INDEXER_K_NORM, ".weight"),
        "engram.wkv.weight": (gguf.MODEL_TENSOR.ENGRAM_KV, ".weight"),
        "main_proj.weight": (gguf.MODEL_TENSOR.DSPARK_MAIN_PROJ, ".weight"),
        "main_norm.weight": (gguf.MODEL_TENSOR.DSPARK_MAIN_NORM, ".weight"),
        "norm.weight": (gguf.MODEL_TENSOR.DSPARK_NORM, ".weight"),
    }
    _ROOT_MAP: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
        "dspark.markov_head.embed.weight": (gguf.MODEL_TENSOR.DSPARK_MARKOV_W1, ".weight"),
        "dspark.markov_head.head.weight": (gguf.MODEL_TENSOR.DSPARK_MARKOV_W2, ".weight"),
        "dspark.confidence_head.proj.weight": (gguf.MODEL_TENSOR.DSPARK_CONF_PROJ, ".weight"),
    }

    def _map_dsv4_tensor_name(self, name: str, bid: int | None) -> tuple[gguf.MODEL_TENSOR, str]:
        if name in self._ROOT_MAP:
            return self._ROOT_MAP[name]
        m = re.match(r"layers\.(\d+)\.(.+)$", name)
        if m is not None and m.group(2) in self._LAYER_MAP:
            return self._LAYER_MAP[m.group(2)]
        key, suffix = super()._map_dsv4_tensor_name(name, bid)
        if key in (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_APE, gguf.MODEL_TENSOR.FFN_GATE_TID2EID) or \
                key.name.startswith(("INDEXER_COMPRESSOR_", "NEXTN_", "HC_HEAD_")):
            raise ValueError(f"{name!r} is a DeepSeek-V4 tensor that V4.1 does not have")
        return key, suffix

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if re.match(r"layers\.\d+\.ffn\.experts\.\d+\.w[123]\.(weight|scale)$", name):
            return []
        if re.match(r"layers\.\d+\.engram\.k_weight$", name):
            return []  # folded into engram_gate together with q_weight
        if (m := re.match(r"layers\.(\d+)\.engram\.q_weight$", name)) is not None:
            k = self.model_tensors.get(f"layers.{m.group(1)}.engram.k_weight")
            if k is None:
                raise KeyError(f"missing layers.{m.group(1)}.engram.k_weight")
            # bf16 * bf16 is exact in f32
            gate = data_torch.float() * k().float()
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENGRAM_GATE, bid), gate)]
        key, suffix = self._map_dsv4_tensor_name(name, bid)
        new_name = self.format_tensor_name(key, None if name.startswith("dspark.") else bid, suffix)
        return [(new_name, data_torch)]

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if new_name.endswith(".engram_gate.weight") or new_name == "conf_proj.weight":
            return gguf.GGMLQuantizationType.F32
        if name in self._dsv4_fp8_dequantized and n_dims >= 2:
            return gguf.GGMLQuantizationType.Q8_0 if self._dense_type == "q8_0" else gguf.GGMLQuantizationType.BF16
        if name in self._dsv4_f32_tensors:
            return gguf.GGMLQuantizationType.F32
        if name in self._dsv4_bf16_tensors and n_dims >= 2:
            return gguf.GGMLQuantizationType.BF16
        return False

