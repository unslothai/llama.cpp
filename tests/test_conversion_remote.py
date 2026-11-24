"""Test the remote (--remote) tensor index with mocked Hugging Face transport; no downloads required.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_conversion_remote.py' -v
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from conversion.base import gguf
from conversion.llama import LlamaModel

SafetensorRemote = gguf.utility.SafetensorRemote


def remote_tensor(url: str) -> gguf.utility.RemoteTensor:
    return gguf.utility.RemoteTensor(dtype="F32", shape=(2,), offset_start=0, size=8, url=url)


class TestRemoteIndex(unittest.TestCase):
    def list_hf_model(self, weight_map, shards):
        index = json.dumps({"weight_map": weight_map}).encode()

        def list_tensors(url):
            return {name: remote_tensor(url) for name in shards[url.rsplit("/", 1)[1]]}

        with mock.patch.object(SafetensorRemote, "check_file_exist", side_effect=lambda url: url.endswith(".index.json")), \
             mock.patch.object(SafetensorRemote, "get_data_by_range", return_value=index), \
             mock.patch.object(SafetensorRemote, "get_list_tensors", side_effect=list_tensors):
            return SafetensorRemote.get_list_tensors_hf_model("org/model")

    def test_sharded(self):
        weight_map = {"a.weight": "model-00001.safetensors", "b.weight": "model-00002.safetensors"}
        shards = {"model-00001.safetensors": ["a.weight"], "model-00002.safetensors": ["b.weight"]}
        tensors = self.list_hf_model(weight_map, shards)
        self.assertEqual(tensors["a.weight"].url.rsplit("/", 1)[1], "model-00001.safetensors")
        self.assertEqual(tensors["b.weight"].url.rsplit("/", 1)[1], "model-00002.safetensors")

    def test_wrong_shard_copy(self):
        weight_map = {"a.weight": "model-00001.safetensors", "b.weight": "model-00002.safetensors"}
        # the later shard holds a stale copy of a.weight
        shards = {"model-00001.safetensors": ["a.weight"], "model-00002.safetensors": ["a.weight", "b.weight"]}
        with self.assertRaisesRegex(ValueError, "wrong-shard"):
            self.list_hf_model(weight_map, shards)

    def test_duplicate_after_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            hparams = {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "num_attention_heads": 1,
            }
            (path / "config.json").write_text(json.dumps(hparams))
            save_file({"model.norm.weight": torch.ones(2)}, str(path / "model.safetensors"))
            model = LlamaModel(path, gguf.LlamaFileType.ALL_F32, path / "out.gguf", hparams=hparams)

            url = "https://huggingface.co/org/model/resolve/main/model.safetensors"
            # both names filter to model.norm.weight
            remote = {"model.norm.weight": remote_tensor(url), "model.language_model.norm.weight": remote_tensor(url)}
            with mock.patch.object(SafetensorRemote, "get_list_tensors_hf_model", return_value=remote):
                with self.assertRaisesRegex(ValueError, "duplicate tensor 'model.norm.weight'"):
                    model.index_tensors(remote_hf_model_id="org/model")

            del remote["model.language_model.norm.weight"]
            with mock.patch.object(SafetensorRemote, "get_list_tensors_hf_model", return_value=remote):
                self.assertEqual(list(model.index_tensors(remote_hf_model_id="org/model")), ["model.norm.weight"])


if __name__ == "__main__":
    unittest.main()
