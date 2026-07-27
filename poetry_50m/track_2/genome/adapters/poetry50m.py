from __future__ import annotations

import gc
import importlib
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import torch
from torch import nn

from genome.adapters.base import Track1Adapter
from genome.hashing import sha256_file, sha256_state_dict
from genome.io import read_json as read_json_file
from genome.io import read_yaml


_SPLIT_DEFAULTS: dict[str, str] = {
    "fit": "train",
    "fingerprint": "train",
    "probe": "validation",
    "development": "validation",
    "validation": "validation",
    "hidden": "test",
    "test": "test",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = read_json_file(path)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"expected a JSON object with string keys: {path}")
    return cast(dict[str, Any], value)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = read_yaml(path)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"expected a YAML mapping with string keys: {path}")
    return cast(dict[str, Any], value)


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("PyTorch with weights_only=True support is required") from error
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"checkpoint must be a mapping with string keys: {path}")
    return cast(Mapping[str, Any], value)


def _state_from_payload(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    for key in ("model", "state_dict", "model_state_dict"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            if any(
                not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
                for name, tensor in value.items()
            ):
                raise TypeError(f"checkpoint field {key!r} is not a tensor state_dict")
            return cast(Mapping[str, torch.Tensor], value)
    if payload and all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in payload.items()
    ):
        return cast(Mapping[str, torch.Tensor], payload)
    raise ValueError("checkpoint contains no supported model state")


def _canonical_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in state.items()
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


class Poetry50MAdapter(Track1Adapter):
    """Concrete GENOME boundary for ``poetry_50m/track_1``.

    This adapter intentionally imports Track 1 lazily. The Track 2 package can therefore be
    installed and tested on its own, while an actual specimen freeze still uses Track 1's exact
    constructor, seeding function, checkpoint schema, tokenizer, and packed evaluation records.
    """

    adapter_id = "poetry50m.track1.gpt-ngpt.v1"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        track1_root: str | Path | None = None,
        model_config: str | Path | None = None,
        train_config: str | Path | None = None,
        prepared_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        initial_snapshot: str | Path | None = None,
        final_snapshot: str | Path | None = None,
        run_manifest: str | Path | None = None,
        train_receipt: str | Path | None = None,
        corpus_manifest: str | Path | None = None,
        split_map: Mapping[str, str] | None = None,
        default_max_batches: int = 16,
        require_complete_endpoint: bool = True,
    ) -> None:
        self.project_config = dict(config or {})
        config_path = self.project_config.get("_config_path")
        self._config_dir = Path(config_path).resolve().parent if config_path else Path.cwd()
        adapter_config = self.project_config.get("adapter", {})
        configured_root = (
            adapter_config.get("project_root")
            if isinstance(adapter_config, Mapping)
            else None
        )
        root_value = track1_root or configured_root or "../track_1"
        self.track1_root = self._resolve_initial(root_value)
        if self.track1_root.name == "src" and (self.track1_root / "poetry50m").is_dir():
            self.track1_src = self.track1_root
            self.track1_root = self.track1_root.parent
        else:
            self.track1_src = self.track1_root / "src"
        self._install_import_path()
        self.poetry50m_import_origin = self._verify_import_origin()

        self.model_config_path = self._resolve_under_root(
            model_config, "configs/model/track1_50m.yaml"
        )
        self.train_config_path = self._resolve_under_root(
            train_config, "configs/training/baseline.yaml"
        )
        self.prepared_dir = self._resolve_under_root(prepared_dir, "artifacts/prepared")
        self.run_dir = self._resolve_under_root(run_dir, "runs/r0")
        self.initial_snapshot_path = self._resolve_under_root(
            initial_snapshot, self.run_dir / "trajectory/initial.pt"
        )
        self.final_snapshot_path = self._resolve_under_root(
            final_snapshot, self.run_dir / "trajectory/final.pt"
        )
        self.run_manifest_path = self._resolve_under_root(
            run_manifest, self.run_dir / "run.manifest.json"
        )
        self.train_receipt_path = self._resolve_under_root(
            train_receipt, self.run_dir / "train.receipt.json"
        )
        self.corpus_manifest_path = self._resolve_under_root(
            corpus_manifest, "artifacts/corpus/manifest.jsonl"
        )
        if (
            isinstance(default_max_batches, bool)
            or not isinstance(default_max_batches, int)
            or default_max_batches < 1
        ):
            raise ValueError("default_max_batches must be a positive integer")
        if not isinstance(require_complete_endpoint, bool):
            raise TypeError("require_complete_endpoint must be a boolean")
        self.default_max_batches = default_max_batches
        self.require_complete_endpoint = require_complete_endpoint
        self.split_map = dict(_SPLIT_DEFAULTS)
        if split_map:
            self.split_map.update({str(key): str(value) for key, value in split_map.items()})
        self._model_config_cache: dict[str, Any] | None = None
        self._train_config_cache: dict[str, Any] | None = None
        self._expected_model_config_hash: str | None = None
        self._expected_train_config_hash: str | None = None

    def _resolve_initial(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self._config_dir / path).resolve()

    def _resolve_under_root(
        self, value: str | Path | None, default: str | Path
    ) -> Path:
        selected = Path(default) if value is None else Path(value)
        selected = selected.expanduser()
        if selected.is_absolute():
            return selected.resolve()
        # Explicit values are written relative to the YAML file, even when the artifact is not
        # present yet (for example R0's still-in-progress final checkpoint). Defaults are rooted
        # at Track 1.
        if value is not None:
            return (self._config_dir / selected).resolve()
        return (self.track1_root / selected).resolve()

    def _install_import_path(self) -> None:
        if not self.track1_src.is_dir():
            raise FileNotFoundError(
                f"Track 1 source directory does not exist: {self.track1_src}. "
                "Set adapter.project_root to poetry_50m/track_1."
            )
        source = str(self.track1_src)
        if source not in sys.path:
            sys.path.insert(0, source)

    def _verify_import_origin(self) -> Path:
        module = importlib.import_module("poetry50m")
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str):
            raise RuntimeError("imported poetry50m package has no concrete file origin")
        origin = Path(raw_origin).resolve()
        try:
            origin.relative_to(self.track1_src.resolve())
        except ValueError as error:
            raise RuntimeError(
                "poetry50m was imported from a different checkout: "
                f"{origin}; expected under {self.track1_src}"
            ) from error
        return origin

    @property
    def model_config_mapping(self) -> dict[str, Any]:
        if self._model_config_cache is None:
            self._model_config_cache = _read_yaml(self.model_config_path)
        return dict(self._model_config_cache)

    @property
    def train_config_mapping(self) -> dict[str, Any]:
        if self._train_config_cache is None:
            self._train_config_cache = _read_yaml(self.train_config_path)
        return dict(self._train_config_cache)

    @property
    def expected_model_config_hash(self) -> str:
        if self._expected_model_config_hash is None:
            from poetry50m.training.engine import mapping_hash

            self._expected_model_config_hash = mapping_hash(self.model_config_mapping)
        return self._expected_model_config_hash

    @property
    def expected_train_config_hash(self) -> str:
        if self._expected_train_config_hash is None:
            from poetry50m.training.engine import mapping_hash

            self._expected_train_config_hash = mapping_hash(self.train_config_mapping)
        return self._expected_train_config_hash

    def _run_manifest(self) -> dict[str, Any] | None:
        return _read_json(self.run_manifest_path) if self.run_manifest_path.is_file() else None

    def build_model(self) -> nn.Module:
        from poetry50m.model import DecoderOnlyTransformer, ModelConfig
        from poetry50m.training.engine import seed_everything

        train = self.train_config_mapping
        seed = train.get("seed")
        deterministic = train.get("deterministic")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Track 1 train config lacks an integer seed")
        if not isinstance(deterministic, bool):
            raise ValueError("Track 1 train config lacks a boolean deterministic setting")
        seed_everything(seed, deterministic)
        return DecoderOnlyTransformer(ModelConfig.from_mapping(self.model_config_mapping))

    def initial_state(self) -> Mapping[str, torch.Tensor]:
        if self.initial_snapshot_path.is_file():
            payload = _safe_torch_load(self.initial_snapshot_path)
            return _canonical_state(_state_from_payload(payload))
        return _canonical_state(self.build_model().state_dict())

    def load_checkpoint(self, model: nn.Module, path: str | Path) -> None:
        payload = _safe_torch_load(Path(path).expanduser().resolve(strict=True))
        state = _state_from_payload(payload)
        missing, unexpected = model.load_state_dict(dict(state), strict=True)
        if missing or unexpected:
            raise ValueError(f"checkpoint mismatch; missing={missing}, unexpected={unexpected}")

    def validate_endpoint_checkpoint(self, path: str | Path) -> dict[str, Any]:
        summary = self.checkpoint_summary(path)
        if self.require_complete_endpoint and not summary["complete"]:
            step = summary.get("global_step")
            maximum = summary.get("max_steps")
            raise ValueError(
                f"R0 is not a complete endpoint: global_step={step}, max_steps={maximum}. "
                "Do not freeze WT until the full Track 1 run has finished."
            )
        for contract in ("model_config_matches", "train_config_matches"):
            if summary.get(contract) is not True:
                raise ValueError(f"R0 endpoint failed its Track 1 {contract} contract")
        if self.run_manifest_path.is_file() and summary.get("run_manifest_matches") is not True:
            raise ValueError("R0 endpoint does not belong to the configured Track 1 run manifest")
        completion = self._completion_artifacts(path, summary)
        summary["completion_artifacts"] = completion
        if self.require_complete_endpoint and completion.get("valid") is not True:
            raise ValueError(
                "R0 completion artifacts failed validation: "
                f"{completion.get('failures', [])}"
            )
        return summary

    def _completion_artifacts(
        self, endpoint: str | Path, endpoint_summary: Mapping[str, Any]
    ) -> dict[str, Any]:
        failures: list[str] = []
        endpoint_path = Path(endpoint).expanduser().resolve(strict=True)
        final_snapshot_summary: dict[str, Any] | None = None
        if self.final_snapshot_path.is_file():
            final_snapshot_summary = self.checkpoint_summary(self.final_snapshot_path)
            for field in (
                "state_sha256",
                "global_step",
                "run_id",
                "model_config_matches",
                "train_config_matches",
                "run_manifest_matches",
            ):
                if final_snapshot_summary.get(field) != endpoint_summary.get(field):
                    failures.append(f"endpoint/final-snapshot mismatch: {field}")
            if final_snapshot_summary.get("complete") is not True:
                failures.append("final trajectory snapshot is not at max_steps")
        else:
            failures.append(f"missing final trajectory snapshot: {self.final_snapshot_path}")

        receipt: dict[str, Any] | None = None
        if self.train_receipt_path.is_file():
            receipt = _read_json(self.train_receipt_path)
            if receipt.get("run_id") != endpoint_summary.get("run_id"):
                failures.append("train receipt run_id does not match endpoint")
            if receipt.get("global_step") != endpoint_summary.get("global_step"):
                failures.append("train receipt global_step does not match endpoint")
            checkpoint_hash = receipt.get("checkpoint_sha256")
            snapshot_hash = receipt.get("snapshot_sha256")
            if not isinstance(checkpoint_hash, str):
                failures.append("train receipt lacks checkpoint_sha256")
            if not isinstance(snapshot_hash, str):
                failures.append("train receipt lacks snapshot_sha256")
            if endpoint_summary.get("checkpoint_kind") == "trainer":
                if checkpoint_hash != sha256_file(endpoint_path):
                    failures.append("train receipt checkpoint hash does not match endpoint")
            elif endpoint_summary.get("checkpoint_kind") == "weight_snapshot":
                if snapshot_hash != sha256_file(endpoint_path):
                    failures.append("train receipt snapshot hash does not match endpoint")
            if self.final_snapshot_path.is_file() and isinstance(snapshot_hash, str):
                if snapshot_hash != sha256_file(self.final_snapshot_path):
                    failures.append("train receipt snapshot hash does not match final snapshot")
            if self.run_manifest_path.is_file():
                declared_manifest_hash = receipt.get("run_manifest_sha256")
                if declared_manifest_hash != sha256_file(self.run_manifest_path):
                    failures.append("train receipt run-manifest hash does not match")
        else:
            failures.append(f"missing train receipt: {self.train_receipt_path}")

        return {
            "valid": not failures,
            "failures": failures,
            "final_snapshot": final_snapshot_summary,
            "train_receipt": receipt,
            "final_snapshot_path": str(self.final_snapshot_path),
            "train_receipt_path": str(self.train_receipt_path),
        }

    def validate_base_checkpoint(
        self, path: str | Path, *, endpoint_checkpoint: str | Path | None = None
    ) -> dict[str, Any]:
        summary = self.checkpoint_summary(path)
        if summary.get("global_step") != 0:
            raise ValueError(
                f"W0 checkpoint must be step 0, got {summary.get('global_step')!r}"
            )
        for contract in ("model_config_matches", "train_config_matches"):
            if summary.get(contract) is not True:
                raise ValueError(f"W0 failed its Track 1 {contract} contract")
        if self.run_manifest_path.is_file() and summary.get("run_manifest_matches") is not True:
            raise ValueError("W0 does not belong to the configured Track 1 run manifest")
        endpoint_run_id: str | None = None
        if endpoint_checkpoint is not None:
            endpoint_summary = self.checkpoint_summary(endpoint_checkpoint)
            endpoint_run_id = cast(str | None, endpoint_summary.get("run_id"))
            base_run_id = cast(str | None, summary.get("run_id"))
            if endpoint_run_id is not None and base_run_id != endpoint_run_id:
                raise ValueError("W0 and WT belong to different Track 1 runs")
        return {
            **summary,
            "valid_base": True,
            "endpoint_run_id": endpoint_run_id,
        }

    def checkpoint_summary(self, path: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(path).expanduser().resolve(strict=True)
        payload = _safe_torch_load(checkpoint_path)
        state = _state_from_payload(payload)
        training_state = payload.get("training_state")
        train_config = payload.get("train_config")
        global_step: int | None = None
        max_steps: int | None = None
        if isinstance(training_state, Mapping):
            candidate = training_state.get("global_step")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                global_step = candidate
        if isinstance(train_config, Mapping):
            candidate = train_config.get("max_steps")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                max_steps = candidate

        # A weights-only snapshot has no trainer state, but its metadata carries the step and
        # exact coordinate identities.
        metadata = payload.get("metadata")
        if global_step is None and isinstance(metadata, Mapping):
            candidate = metadata.get("step")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                global_step = candidate
        if max_steps is None:
            candidate = self.train_config_mapping.get("max_steps")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                max_steps = candidate

        checkpoint_kind = (
            "trainer"
            if payload.get("format_version") == 2 and "model" in payload
            else "weight_snapshot"
            if payload.get("format") == "poetry50m.weights.v1"
            else "state_dict"
        )

        declared_model_hash = payload.get("model_config_hash")
        declared_train_hash = payload.get("train_config_hash")
        if isinstance(metadata, Mapping):
            declared_model_hash = metadata.get("model_config_hash", declared_model_hash)
            declared_train_hash = metadata.get("training_config_hash", declared_train_hash)

        from poetry50m.training.engine import mapping_hash

        embedded_model_config = payload.get("model_config")
        embedded_train_config = payload.get("train_config")
        if not isinstance(declared_model_hash, str) and isinstance(
            embedded_model_config, Mapping
        ):
            declared_model_hash = mapping_hash(
                cast(Mapping[str, Any], embedded_model_config)
            )
        if not isinstance(declared_train_hash, str) and isinstance(
            embedded_train_config, Mapping
        ):
            declared_train_hash = mapping_hash(
                cast(Mapping[str, Any], embedded_train_config)
            )
        model_config_matches = (
            declared_model_hash == self.expected_model_config_hash
            if isinstance(declared_model_hash, str)
            else None
        )
        train_config_matches = (
            declared_train_hash == self.expected_train_config_hash
            if isinstance(declared_train_hash, str)
            else None
        )

        run_id: str | None = None
        if isinstance(metadata, Mapping) and isinstance(metadata.get("run_id"), str):
            run_id = cast(str, metadata["run_id"])
        run_metadata = payload.get("run_metadata")
        if (
            run_id is None
            and isinstance(run_metadata, Mapping)
            and isinstance(run_metadata.get("run_id"), str)
        ):
            run_id = cast(str, run_metadata["run_id"])
        run_manifest = self._run_manifest()
        expected_run_id = (
            cast(str, run_manifest["run_id"])
            if isinstance(run_manifest, Mapping)
            and isinstance(run_manifest.get("run_id"), str)
            else None
        )
        run_manifest_matches = (
            run_id == expected_run_id
            if run_id is not None and expected_run_id is not None
            else None
        )

        complete = (
            global_step is not None
            and max_steps is not None
            and global_step >= max_steps
        )
        return {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "format": payload.get("format", payload.get("format_version")),
            "checkpoint_kind": checkpoint_kind,
            "global_step": global_step,
            "max_steps": max_steps,
            "progress_fraction": (
                None
                if global_step is None or max_steps in (None, 0)
                else min(global_step / max_steps, 1.0)
            ),
            "complete": complete,
            "state_tensor_count": len(state),
            "state_numel": sum(tensor.numel() for tensor in state.values()),
            "state_sha256": sha256_state_dict(state),
            "declared_model_config_hash": declared_model_hash,
            "declared_train_config_hash": declared_train_hash,
            "expected_model_config_hash": self.expected_model_config_hash,
            "expected_train_config_hash": self.expected_train_config_hash,
            "model_config_matches": model_config_matches,
            "train_config_matches": train_config_matches,
            "run_id": run_id,
            "expected_run_id": expected_run_id,
            "run_manifest_matches": run_manifest_matches,
            "run_identity": payload.get("run_identity"),
        }

    def architecture_manifest(self, model: nn.Module) -> dict[str, Any]:
        result = super().architecture_manifest(model)
        model_config = getattr(model, "config", None)
        if model_config is not None:
            try:
                result["model_config"] = asdict(model_config)
            except TypeError:
                result["model_config"] = self.model_config_mapping
        result.update(
            {
                "track1_root": str(self.track1_root),
                "poetry50m_import_origin": str(self.poetry50m_import_origin),
                "model_config_file": _file_record(self.model_config_path),
                "train_config_file": _file_record(self.train_config_path),
                "architecture": self.model_config_mapping.get("architecture"),
            }
        )
        source_files = [
            self.track1_src / "poetry50m/model/config.py",
            self.track1_src / "poetry50m/model/transformer.py",
        ]
        result["coordinate_source_files"] = [
            _file_record(path) for path in source_files if path.is_file()
        ]
        return result

    def tokenizer_manifest(self) -> dict[str, Any]:
        tokenizer_path = self.prepared_dir / "tokenizer.json"
        metadata_path = self.prepared_dir / "metadata.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"prepared tokenizer does not exist: {tokenizer_path}")
        from tokenizers import Tokenizer
        from poetry50m.data.tokenizer import SPECIAL_TOKENS

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
        result: dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "tokenizer_file": _file_record(tokenizer_path),
            "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
            "special_tokens": special_ids,
        }
        if metadata_path.is_file():
            metadata = _read_json(metadata_path)
            result["prepared_metadata_file"] = _file_record(metadata_path)
            result["declared_tokenizer_hash"] = metadata.get("tokenizer_hash")
        return result

    def corpus_manifest(self) -> dict[str, Any]:
        metadata_path = self.prepared_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"prepared metadata does not exist: {metadata_path}")
        metadata = _read_json(metadata_path)
        result: dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "prepared_root": str(self.prepared_dir),
            "prepared_metadata_file": _file_record(metadata_path),
            "prepared_metadata": metadata,
        }
        if self.corpus_manifest_path.is_file():
            result["source_document_manifest"] = _file_record(self.corpus_manifest_path)
        return result

    def training_recipe(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "model_config": self.model_config_mapping,
            "train_config": self.train_config_mapping,
            "model_config_file": _file_record(self.model_config_path),
            "train_config_file": _file_record(self.train_config_path),
            "run_directory": str(self.run_dir),
        }
        if self.run_manifest_path.is_file():
            result["run_manifest_file"] = _file_record(self.run_manifest_path)
            result["run_manifest"] = _read_json(self.run_manifest_path)
        telemetry = self.run_dir / "telemetry.jsonl"
        if telemetry.is_file():
            result["telemetry_file"] = _file_record(telemetry)
        receipt = self.run_dir / "train.receipt.json"
        if receipt.is_file():
            result["train_receipt_file"] = _file_record(receipt)
            result["train_receipt"] = _read_json(receipt)
        return result

    def split_manifest(self) -> dict[str, Any]:
        metadata_path = self.prepared_dir / "metadata.json"
        metadata = _read_json(metadata_path)
        packed: dict[str, Any] = {}
        for split in ("train", "validation", "test"):
            path = self.prepared_dir / f"{split}.packed.jsonl"
            if path.is_file():
                packed[split] = _file_record(path)
        split_fields = {
            key: value
            for key, value in metadata.items()
            if "split" in key.lower() or key in {"counts", "file_hashes", "input_hashes"}
        }
        return {
            "adapter_id": self.adapter_id,
            "semantic_to_track1": dict(sorted(self.split_map.items())),
            "packed_files": packed,
            "prepared_metadata_sha256": sha256_file(metadata_path),
            "prepared_split_fields": split_fields,
        }

    def _track1_split(self, split: str) -> str:
        track1_split = self.split_map.get(split, split)
        if track1_split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported Track 1 evaluation split: {split!r} -> {track1_split!r}")
        return track1_split

    def evaluation_batches(self, split: str, max_batches: int | None = None) -> Iterable[Any]:
        from poetry50m.data.artifacts import read_packed_sequences

        track1_split = self._track1_split(split)
        packs_path = self.prepared_dir / f"{track1_split}.packed.jsonl"
        packs: Sequence[Any] = read_packed_sequences(packs_path)
        if not packs:
            raise ValueError(f"prepared artifact has no {track1_split} packs")
        limit = len(packs) if max_batches is None else max_batches
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("evaluation max_batches must be a positive integer or null for all")
        for pack in packs[:limit]:
            yield {
                "input_ids": torch.tensor([pack.input_ids[:-1]], dtype=torch.long),
                "targets": torch.tensor([pack.input_ids[1:]], dtype=torch.long),
                "loss_mask": torch.tensor([pack.loss_mask[1:]], dtype=torch.bool),
                "example_ids": (f"{pack.objective}:pack:{pack.pack_id}",),
                "data_token_count": max(0, len(pack.input_ids) - 1),
            }

    def model_call(self, batch: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not isinstance(batch, Mapping) or "input_ids" not in batch:
            raise TypeError("poetry50m evaluation batches must be mappings with input_ids")
        return (batch["input_ids"],), {}

    def loss_from_logits(self, logits: torch.Tensor, batch: Any) -> tuple[torch.Tensor, int]:
        if not isinstance(batch, Mapping):
            raise TypeError("poetry50m evaluation batches must be mappings")
        targets = batch.get("targets")
        loss_mask = batch.get("loss_mask")
        if not isinstance(targets, torch.Tensor) or targets.shape != logits.shape[:2]:
            raise ValueError("poetry50m targets must match the logit batch/sequence shape")
        ignore_index = self.model_config_mapping.get("ignore_index", -100)
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
            raise ValueError("Track 1 model config ignore_index must be an integer")
        if loss_mask is None:
            valid = targets.ne(ignore_index)
            weights = valid.to(dtype=logits.dtype)
        elif isinstance(loss_mask, torch.Tensor) and loss_mask.shape == targets.shape:
            valid = targets.ne(ignore_index) & loss_mask.to(dtype=torch.bool)
            weights = loss_mask.to(dtype=logits.dtype) * valid.to(dtype=logits.dtype)
        else:
            raise ValueError("poetry50m loss_mask must match targets")
        count = int(valid.sum().item())
        if count < 1:
            raise ValueError("poetry50m batch contains no supervised targets")
        token_loss = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1),
            targets.flatten(),
            ignore_index=ignore_index,
            reduction="none",
        ).view_as(targets)
        return (token_loss * weights.to(dtype=token_loss.dtype)).sum(), count

    def batch_loss(self, model: nn.Module, batch: Any) -> tuple[torch.Tensor, int]:
        if not isinstance(batch, Mapping):
            raise TypeError("poetry50m evaluation batches must be mappings")
        output = model(
            batch["input_ids"],
            targets=batch["targets"],
            loss_mask=batch.get("loss_mask"),
        )
        loss = getattr(output, "loss", None)
        count = getattr(output, "token_count", None)
        if (
            not isinstance(loss, torch.Tensor)
            or isinstance(count, bool)
            or not isinstance(count, int)
        ):
            raise TypeError("poetry50m model did not return loss and token_count")
        if count < 1:
            raise ValueError("poetry50m batch contains no supervised targets")
        return loss * count, count

    def export_evaluation_checkpoint(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        template_checkpoint: str | Path,
        output: str | Path,
        candidate_id: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a full Track 1 trainer checkpoint for read-only generation/evaluation.

        The optimizer, scheduler, RNG, and stream records remain the R0 template so Track 1's
        strict loader accepts the file. This artifact is explicitly marked evaluation-only and
        must never be used to resume training.
        """
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        template_path = Path(template_checkpoint).expanduser().resolve(strict=True)
        template = _safe_torch_load(template_path)
        if template.get("format_version") != 2 or not isinstance(template.get("model"), Mapping):
            raise ValueError(
                "Track 1 evaluation export requires a format_version 2 trainer checkpoint"
            )
        template_state = _state_from_payload(template)
        if tuple(state) != tuple(template_state):
            raise ValueError("candidate state names/order do not match the Track 1 template")
        clean_state: dict[str, torch.Tensor] = {}
        for name, template_tensor in template_state.items():
            candidate = state[name]
            if candidate.shape != template_tensor.shape or candidate.dtype != template_tensor.dtype:
                raise ValueError(f"candidate tensor contract differs for {name}")
            if not torch.isfinite(candidate).all():
                raise ValueError(f"candidate tensor is non-finite: {name}")
            clean_state[name] = candidate.detach().cpu().contiguous()

        exported = dict(template)
        exported["model"] = clean_state
        exported["genome_evaluation"] = {
            "format_version": 1,
            "evaluation_only": True,
            "resume_forbidden": True,
            "candidate_id": candidate_id,
            "candidate_state_sha256": sha256_state_dict(clean_state),
            "template_checkpoint_sha256": sha256_file(template_path),
            "provenance": dict(provenance or {}),
        }
        destination = Path(output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(exported, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        verified = _safe_torch_load(destination)
        verified_state = _state_from_payload(verified)
        state_hash = sha256_state_dict(verified_state)
        if state_hash != exported["genome_evaluation"]["candidate_state_sha256"]:
            destination.unlink(missing_ok=True)
            raise ValueError("exported Track 1 evaluation checkpoint failed state verification")
        return {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
            "candidate_id": candidate_id,
            "candidate_state_sha256": state_hash,
            "template_checkpoint_sha256": sha256_file(template_path),
            "evaluation_only": True,
            "resume_forbidden": True,
        }

    def latest_checkpoint(self) -> Path | None:
        checkpoint_dir = self.run_dir / "checkpoints"
        if not checkpoint_dir.is_dir():
            return None
        preferred = checkpoint_dir / "final.pt"
        if preferred.is_file():
            return preferred
        candidates = sorted(
            (path for path in checkpoint_dir.glob("*.pt") if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        return candidates[0] if candidates else None

    def preflight(self, endpoint: str | Path | None = None) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        model_a = self.build_model().cpu()
        hash_a = sha256_state_dict(model_a.state_dict())
        parameter_count = sum(parameter.numel() for parameter in model_a.parameters())
        del model_a
        gc.collect()
        model_b = self.build_model().cpu()
        hash_b = sha256_state_dict(model_b.state_dict())
        del model_b
        gc.collect()
        checks["W0_reproducible"] = hash_a == hash_b
        checks["W0_sha256"] = hash_a
        checks["parameter_count"] = parameter_count
        checks["expected_parameter_count"] = (
            50_343_424 if self.model_config_mapping.get("architecture") == "gpt" else 54_596_096
        )
        checks["parameter_count_matches_known_track1_shape"] = (
            checks["parameter_count"] == checks["expected_parameter_count"]
        )
        checks["paths"] = {
            "track1_root": str(self.track1_root),
            "poetry50m_import_origin": str(self.poetry50m_import_origin),
            "model_config": str(self.model_config_path),
            "train_config": str(self.train_config_path),
            "prepared_dir": str(self.prepared_dir),
            "run_dir": str(self.run_dir),
            "initial_snapshot": str(self.initial_snapshot_path),
            "final_snapshot": str(self.final_snapshot_path),
            "run_manifest": str(self.run_manifest_path),
            "train_receipt": str(self.train_receipt_path),
        }
        required_files = {
            "model_config": self.model_config_path,
            "train_config": self.train_config_path,
            "prepared_metadata": self.prepared_dir / "metadata.json",
            "tokenizer": self.prepared_dir / "tokenizer.json",
            "validation_packs": self.prepared_dir / "validation.packed.jsonl",
            "test_packs": self.prepared_dir / "test.packed.jsonl",
            "initial_snapshot": self.initial_snapshot_path,
            "final_snapshot": self.final_snapshot_path,
            "run_manifest": self.run_manifest_path,
            "train_receipt": self.train_receipt_path,
        }
        checks["files"] = {name: path.is_file() for name, path in required_files.items()}
        selected = (
            Path(endpoint).expanduser().resolve()
            if endpoint is not None
            else self.latest_checkpoint()
        )
        if self.initial_snapshot_path.is_file():
            try:
                checks["base"] = self.validate_base_checkpoint(
                    self.initial_snapshot_path,
                    endpoint_checkpoint=(
                        selected if selected is not None and selected.is_file() else None
                    ),
                )
            except (OSError, TypeError, ValueError) as error:
                checks["base"] = {"valid_base": False, "error": str(error)}
        else:
            checks["base"] = {
                "valid_base": False,
                "error": "initial snapshot is missing",
            }
        if selected is not None and selected.is_file():
            endpoint_summary = self.checkpoint_summary(selected)
            endpoint_summary["completion_artifacts"] = self._completion_artifacts(
                selected, endpoint_summary
            )
            checks["endpoint"] = endpoint_summary
        else:
            checks["endpoint"] = {
                "path": None if selected is None else str(selected),
                "complete": False,
            }
        endpoint_summary = checks["endpoint"]
        checks["ready_to_freeze"] = bool(
            checks["W0_reproducible"]
            and checks["parameter_count_matches_known_track1_shape"]
            and all(checks["files"].values())
            and checks["base"].get("valid_base")
            and endpoint_summary.get("complete")
            and endpoint_summary.get("model_config_matches") is True
            and endpoint_summary.get("train_config_matches") is True
            and endpoint_summary.get("run_manifest_matches") is True
            and endpoint_summary.get("completion_artifacts", {}).get("valid") is True
        )
        return checks


def create_adapter(
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Poetry50MAdapter:
    return Poetry50MAdapter(config=config, **kwargs)
