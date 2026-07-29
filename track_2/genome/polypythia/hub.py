from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from ..hashing import sha256_file, sha256_json
from ..io import ensure_dir, read_json, write_json
from .catalog import PolyPythiaLife, RoundOneCatalog

_WEIGHT_PREFERENCE = ("model.safetensors", "pytorch_model.bin")
_TOKENIZER_FILES = {
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


@dataclass(frozen=True)
class HubFile:
    name: str
    size: int
    sha256: str | None
    git_blob_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError(f"Hub file must be a direct child: {self.name!r}")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError(f"invalid Hub file size: {self.size!r}")
        if self.sha256 is None and self.git_blob_id is None:
            raise ValueError(f"Hub file lacks an integrity identity: {self.name}")
        if self.sha256 is not None:
            _validate_sha256(self.sha256, field=f"{self.name}.sha256")
        if self.git_blob_id is not None:
            _validate_hex(self.git_blob_id, length=40, field=f"{self.name}.git_blob_id")


@dataclass(frozen=True)
class CheckpointSource:
    step: int
    branch: str
    commit: str
    weight: HubFile


@dataclass(frozen=True)
class LifeSourcePlan:
    run_id: str
    seed: int
    data_order_seed: int
    repository: str
    split: str
    main_commit: str
    checkpoints: tuple[CheckpointSource, ...]


@dataclass(frozen=True)
class DatasetOrderPlan:
    repository: str
    commit: str
    seed_files: dict[str, tuple[HubFile, ...]]


@dataclass(frozen=True)
class TokenizerSourcePlan:
    repository: str
    commit: str
    files: tuple[HubFile, ...]


@dataclass(frozen=True)
class RoundOneSourcePlan:
    catalog: dict[str, Any]
    lives: tuple[LifeSourcePlan, ...]
    dataset_order: DatasetOrderPlan
    tokenizer: TokenizerSourcePlan
    catalogued_checkpoint_bytes: int
    sealed_materialization_bytes: int
    revealed_materialization_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "catalogued_checkpoint_bytes",
            "sealed_materialization_bytes",
            "revealed_materialization_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not (
            self.sealed_materialization_bytes
            <= self.revealed_materialization_bytes
            <= self.catalogued_checkpoint_bytes
        ):
            raise ValueError("source-plan materialization byte totals are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "format": "GENOME_POLYPYTHIA_SOURCE_PLAN",
            "version": "0.2.0",
            "catalog": self.catalog,
            "lives": [
                {
                    **{key: item for key, item in asdict(life).items() if key != "checkpoints"},
                    "checkpoints": [
                        {
                            "step": checkpoint.step,
                            "branch": checkpoint.branch,
                            "commit": checkpoint.commit,
                            "weight": asdict(checkpoint.weight),
                        }
                        for checkpoint in life.checkpoints
                    ],
                }
                for life in self.lives
            ],
            "dataset_order": {
                "repository": self.dataset_order.repository,
                "commit": self.dataset_order.commit,
                "seed_files": {
                    seed: [asdict(file) for file in files]
                    for seed, files in sorted(self.dataset_order.seed_files.items())
                },
            },
            "tokenizer": {
                "repository": self.tokenizer.repository,
                "commit": self.tokenizer.commit,
                "files": [asdict(file) for file in self.tokenizer.files],
            },
            "catalogued_checkpoint_bytes": self.catalogued_checkpoint_bytes,
            "materialization": {
                "primary_checkpoint_policy": "W0_and_WT_only",
                "intermediate_checkpoints": "pinned_but_not_downloaded_for_primary_experiment",
                "sealed_bytes": self.sealed_materialization_bytes,
                "revealed_bytes": self.revealed_materialization_bytes,
            },
            "endpoint_access": {
                "training": "available_to_decoder_and_compiler_training",
                "development": "available_for_model_selection_only",
                "hidden": "step0_only_until_prediction_seal",
            },
        }
        value["content_sha256"] = sha256_json(value)
        return value


def _validate_sha256(value: object, *, field: str) -> str:
    return _validate_hex(value, length=64, field=field)


def _validate_hex(value: object, *, length: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a {length}-character lowercase hexadecimal digest")
    return value


def _validate_commit(value: object, *, field: str) -> str:
    return _validate_hex(value, length=40, field=field)


def _integrity_identity(
    sibling: Any,
    *,
    repository: str,
    revision: str,
) -> tuple[str | None, str | None]:
    lfs = getattr(sibling, "lfs", None)
    sha256 = getattr(lfs, "sha256", None)
    if sha256 is None and isinstance(lfs, Mapping):
        sha256 = lfs.get("sha256")
    field = f"{repository}@{revision}:{getattr(sibling, 'rfilename', '?')}"
    if sha256 is not None:
        return _validate_sha256(sha256, field=field), None
    blob_id = getattr(sibling, "blob_id", None)
    return None, _validate_hex(blob_id, length=40, field=field)


def _hub_file(sibling: Any, *, repository: str, revision: str) -> HubFile:
    name = getattr(sibling, "rfilename", None)
    size = getattr(sibling, "size", None)
    if not isinstance(name, str):
        raise TypeError(f"Hub returned an invalid filename for {repository}@{revision}")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise TypeError(f"Hub returned an invalid size for {repository}@{revision}:{name}")
    sha256, git_blob_id = _integrity_identity(
        sibling,
        repository=repository,
        revision=revision,
    )
    return HubFile(name=name, size=size, sha256=sha256, git_blob_id=git_blob_id)


def _resolve_lfs_weight(
    client: httpx.Client,
    *,
    endpoint: str,
    repository: str,
    commit: str,
    filename: str,
) -> HubFile | None:
    url = f"{endpoint}/{repository}/resolve/{commit}/{filename}"
    for attempt in range(5):
        response = client.head(url)
        if response.status_code == 404:
            return None
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 4:
                response.raise_for_status()
            raw_delay = response.headers.get("retry-after", "1")
            try:
                delay = float(raw_delay)
            except ValueError:
                delay = 1.0
            time.sleep(min(max(delay, 0.1), 30.0))
            continue
        if response.status_code not in {200, 302, 303, 307, 308}:
            response.raise_for_status()
        resolved_commit = _validate_commit(
            response.headers.get("x-repo-commit"),
            field=f"{repository}@{commit}:{filename}.resolved_commit",
        )
        if resolved_commit != commit:
            raise ValueError(f"Hub resolved {repository}@{commit}:{filename} to another commit")
        raw_size = response.headers.get("x-linked-size")
        raw_sha256 = response.headers.get("x-linked-etag", "").strip('"')
        try:
            size = int(raw_size) if raw_size is not None else -1
        except ValueError as error:
            raise ValueError(
                f"Hub returned an invalid LFS size for {repository}@{commit}:{filename}"
            ) from error
        return HubFile(
            name=filename,
            size=size,
            sha256=_validate_sha256(
                raw_sha256,
                field=f"{repository}@{commit}:{filename}.sha256",
            ),
        )
    raise RuntimeError("unreachable LFS resolver retry state")


def _select_resolved_weight(
    client: httpx.Client,
    *,
    endpoint: str,
    repository: str,
    commit: str,
) -> HubFile:
    for filename in _WEIGHT_PREFERENCE:
        weight = _resolve_lfs_weight(
            client,
            endpoint=endpoint,
            repository=repository,
            commit=commit,
            filename=filename,
        )
        if weight is not None:
            return weight
    raise ValueError(f"{repository}@{commit} does not contain a supported weight file")


def _ref_map(api: Any, life: PolyPythiaLife) -> tuple[str, dict[str, str]]:
    refs = api.list_repo_refs(life.repository, repo_type="model")
    branches = {
        branch.name: _validate_commit(
            branch.target_commit,
            field=f"{life.repository}:{branch.name}",
        )
        for branch in refs.branches
    }
    if "main" not in branches:
        raise ValueError(f"{life.repository} has no main branch")
    return branches["main"], branches


def _checkpoint_metadata(
    client: httpx.Client,
    *,
    endpoint: str,
    repository: str,
    step: int,
    branch: str,
    commit: str,
    filename: str,
) -> CheckpointSource:
    weight = _resolve_lfs_weight(
        client,
        endpoint=endpoint,
        repository=repository,
        commit=commit,
        filename=filename,
    )
    if weight is None:
        raise ValueError(f"{repository}@{branch} lacks the selected weight file {filename}")
    return CheckpointSource(
        step=step,
        branch=branch,
        commit=commit,
        weight=weight,
    )


def _plan_life(api: Any, catalog: RoundOneCatalog, life: PolyPythiaLife) -> LifeSourcePlan:
    main_commit, branches = _ref_map(api, life)
    required = {catalog.checkpoints.branch(step): step for step in catalog.checkpoints.steps}
    missing = sorted(set(required) - set(branches))
    if missing:
        raise ValueError(f"{life.repository} lacks required checkpoint branches: {missing}")
    endpoint = str(getattr(api, "endpoint", "https://huggingface.co")).rstrip("/")
    initial_branch = catalog.checkpoints.branch(catalog.checkpoints.initial_step)
    with httpx.Client(follow_redirects=False, timeout=30.0) as client:
        selected_weight = _select_resolved_weight(
            client,
            endpoint=endpoint,
            repository=life.repository,
            commit=branches[initial_branch],
        )
        checkpoints: dict[int, CheckpointSource] = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    _checkpoint_metadata,
                    client,
                    endpoint=endpoint,
                    repository=life.repository,
                    step=step,
                    branch=branch,
                    commit=branches[branch],
                    filename=selected_weight.name,
                ): step
                for branch, step in required.items()
            }
            for future in as_completed(futures):
                step = futures[future]
                checkpoints[step] = future.result()
    return LifeSourcePlan(
        run_id=life.run_id,
        seed=life.seed,
        data_order_seed=life.data_order_seed,
        repository=life.repository,
        split=life.split,
        main_commit=main_commit,
        checkpoints=tuple(checkpoints[step] for step in catalog.checkpoints.steps),
    )


def _plan_dataset_order(api: Any, catalog: RoundOneCatalog) -> DatasetOrderPlan:
    info = api.dataset_info(catalog.dataset_repository, files_metadata=True)
    commit = _validate_commit(info.sha, field=f"{catalog.dataset_repository}.commit")
    files_by_seed: dict[str, list[HubFile]] = {
        str(life.data_order_seed): [] for life in catalog.lives
    }
    for sibling in info.siblings or ():
        filename = getattr(sibling, "rfilename", "")
        for seed, seed_files in files_by_seed.items():
            prefix = f"seed{seed}/"
            if filename.startswith(prefix):
                relative = filename[len(prefix) :]
                size = getattr(sibling, "size", None)
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise TypeError(
                        f"Hub returned an invalid size for "
                        f"{catalog.dataset_repository}@{commit}:{filename}"
                    )
                sha256, git_blob_id = _integrity_identity(
                    sibling,
                    repository=catalog.dataset_repository,
                    revision=commit,
                )
                seed_files.append(
                    HubFile(
                        name=relative,
                        size=size,
                        sha256=sha256,
                        git_blob_id=git_blob_id,
                    )
                )
                break
    for seed, files in files_by_seed.items():
        if len(files) != 3:
            raise ValueError(
                f"{catalog.dataset_repository} must expose three order maps for seed {seed}; "
                f"found {len(files)}"
            )
    return DatasetOrderPlan(
        repository=catalog.dataset_repository,
        commit=commit,
        seed_files={
            seed: tuple(sorted(files, key=lambda file: file.name))
            for seed, files in files_by_seed.items()
        },
    )


def _plan_tokenizer(api: Any, catalog: RoundOneCatalog) -> TokenizerSourcePlan:
    info = api.model_info(catalog.tokenizer_source, files_metadata=True)
    commit = _validate_commit(info.sha, field=f"{catalog.tokenizer_source}.commit")
    files = []
    for sibling in info.siblings or ():
        filename = getattr(sibling, "rfilename", "")
        if filename in _TOKENIZER_FILES:
            files.append(
                _hub_file(
                    sibling,
                    repository=catalog.tokenizer_source,
                    revision=commit,
                )
            )
    present = {file.name for file in files}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(present):
        raise ValueError(f"tokenizer source lacks required files: {sorted(required - present)}")
    return TokenizerSourcePlan(
        repository=catalog.tokenizer_source,
        commit=commit,
        files=tuple(sorted(files, key=lambda file: file.name)),
    )


def build_source_plan(
    catalog: RoundOneCatalog,
    *,
    api: Any | None = None,
) -> RoundOneSourcePlan:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    lives = tuple(_plan_life(api, catalog, life) for life in catalog.lives)
    catalogued_checkpoint_bytes = sum(
        checkpoint.weight.size for life in lives for checkpoint in life.checkpoints
    )
    if catalogued_checkpoint_bytes <= 0 or not math.isfinite(float(catalogued_checkpoint_bytes)):
        raise ValueError("source plan has an invalid checkpoint byte total")
    initial_step = catalog.checkpoints.initial_step
    final_step = catalog.checkpoints.final_step
    endpoint_weights = {
        life.run_id: {
            checkpoint.step: checkpoint.weight.size
            for checkpoint in life.checkpoints
            if checkpoint.step in {initial_step, final_step}
        }
        for life in lives
    }
    revealed_materialization_bytes = sum(
        weights[initial_step] + weights[final_step] for weights in endpoint_weights.values()
    )
    sealed_materialization_bytes = sum(
        weights[initial_step] + (0 if life.split == "hidden" else weights[final_step])
        for life in lives
        for weights in (endpoint_weights[life.run_id],)
    )
    return RoundOneSourcePlan(
        catalog=catalog.to_dict(),
        lives=lives,
        dataset_order=_plan_dataset_order(api, catalog),
        tokenizer=_plan_tokenizer(api, catalog),
        catalogued_checkpoint_bytes=catalogued_checkpoint_bytes,
        sealed_materialization_bytes=sealed_materialization_bytes,
        revealed_materialization_bytes=revealed_materialization_bytes,
    )


def save_source_plan(plan: RoundOneSourcePlan, path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, plan.to_dict(), canonical=True)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "catalogued_checkpoint_bytes": plan.catalogued_checkpoint_bytes,
        "sealed_materialization_bytes": plan.sealed_materialization_bytes,
        "revealed_materialization_bytes": plan.revealed_materialization_bytes,
    }


def _hub_file_from_dict(value: Mapping[str, Any]) -> HubFile:
    return HubFile(
        name=str(value["name"]),
        size=int(value["size"]),
        sha256=(None if value.get("sha256") is None else str(value["sha256"])),
        git_blob_id=(None if value.get("git_blob_id") is None else str(value["git_blob_id"])),
    )


def load_source_plan(path: str | Path) -> RoundOneSourcePlan:
    raw = read_json(path)
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise TypeError("source plan must be an object with string keys")
    if raw.get("format") != "GENOME_POLYPYTHIA_SOURCE_PLAN":
        raise ValueError("not a PolyPythia source plan")
    if raw.get("version") != "0.2.0":
        raise ValueError(f"unsupported PolyPythia source plan version: {raw.get('version')!r}")
    declared_hash = _validate_sha256(raw.get("content_sha256"), field="content_sha256")
    content = dict(raw)
    content.pop("content_sha256", None)
    if sha256_json(content) != declared_hash:
        raise ValueError("PolyPythia source plan content hash mismatch")
    raw_lives = raw.get("lives")
    if not isinstance(raw_lives, Sequence) or isinstance(raw_lives, (str, bytes)):
        raise TypeError("source plan lives must be a sequence")
    lives = []
    for raw_life in raw_lives:
        if not isinstance(raw_life, Mapping):
            raise TypeError("source plan life must be a mapping")
        raw_checkpoints = raw_life.get("checkpoints")
        if not isinstance(raw_checkpoints, Sequence) or isinstance(raw_checkpoints, (str, bytes)):
            raise TypeError("source plan checkpoints must be a sequence")
        checkpoints = []
        for checkpoint in raw_checkpoints:
            if not isinstance(checkpoint, Mapping) or not isinstance(
                checkpoint.get("weight"), Mapping
            ):
                raise TypeError("source plan checkpoint must declare a weight mapping")
            checkpoints.append(
                CheckpointSource(
                    step=int(checkpoint["step"]),
                    branch=str(checkpoint["branch"]),
                    commit=_validate_commit(
                        checkpoint["commit"],
                        field=f"{raw_life.get('run_id')}.checkpoint.commit",
                    ),
                    weight=_hub_file_from_dict(checkpoint["weight"]),
                )
            )
        lives.append(
            LifeSourcePlan(
                run_id=str(raw_life["run_id"]),
                seed=int(raw_life["seed"]),
                data_order_seed=int(raw_life["data_order_seed"]),
                repository=str(raw_life["repository"]),
                split=str(raw_life["split"]),
                main_commit=_validate_commit(
                    raw_life["main_commit"],
                    field=f"{raw_life.get('run_id')}.main_commit",
                ),
                checkpoints=tuple(checkpoints),
            )
        )
    raw_dataset = raw.get("dataset_order")
    if not isinstance(raw_dataset, Mapping) or not isinstance(
        raw_dataset.get("seed_files"), Mapping
    ):
        raise TypeError("source plan dataset_order is invalid")
    dataset = DatasetOrderPlan(
        repository=str(raw_dataset["repository"]),
        commit=_validate_commit(raw_dataset["commit"], field="dataset_order.commit"),
        seed_files={
            str(seed): tuple(_hub_file_from_dict(file) for file in files)
            for seed, files in raw_dataset["seed_files"].items()
        },
    )
    raw_tokenizer = raw.get("tokenizer")
    if not isinstance(raw_tokenizer, Mapping):
        raise TypeError("source plan tokenizer is invalid")
    materialization = raw.get("materialization")
    if not isinstance(materialization, Mapping):
        raise TypeError("source plan materialization is invalid")
    plan = RoundOneSourcePlan(
        catalog=dict(raw["catalog"]),
        lives=tuple(lives),
        dataset_order=dataset,
        tokenizer=TokenizerSourcePlan(
            repository=str(raw_tokenizer["repository"]),
            commit=_validate_commit(raw_tokenizer["commit"], field="tokenizer.commit"),
            files=tuple(_hub_file_from_dict(file) for file in raw_tokenizer["files"]),
        ),
        catalogued_checkpoint_bytes=int(raw["catalogued_checkpoint_bytes"]),
        sealed_materialization_bytes=int(materialization["sealed_bytes"]),
        revealed_materialization_bytes=int(materialization["revealed_bytes"]),
    )
    if plan.to_dict()["content_sha256"] != declared_hash:
        raise ValueError("source plan does not round-trip canonically")
    return plan


def iter_materializable_checkpoints(
    plan: RoundOneSourcePlan,
    *,
    splits: Iterable[str],
    reveal_hidden: bool,
) -> Iterable[tuple[LifeSourcePlan, CheckpointSource]]:
    selected_splits = set(splits)
    valid_splits = {"training", "development", "hidden"}
    if not selected_splits or not selected_splits.issubset(valid_splits):
        raise ValueError(
            f"splits must be a non-empty subset of {sorted(valid_splits)}; "
            f"got {sorted(selected_splits)}"
        )
    initial_step = int(plan.catalog["initial_step"])
    final_step = int(plan.catalog["final_step"])
    for life in plan.lives:
        if life.split not in selected_splits:
            continue
        for checkpoint in life.checkpoints:
            allowed_steps = {initial_step, final_step}
            if life.split == "hidden" and not reveal_hidden:
                allowed_steps = {initial_step}
            if checkpoint.step not in allowed_steps:
                continue
            yield life, checkpoint


def _validate_prediction_seal(
    path: str | Path,
    *,
    hidden_run_id: str,
    source_plan_content_sha256: str,
) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise TypeError("prediction seal must be an object")
    if value.get("format") != "GENOME_HIDDEN_PREDICTION_SEAL":
        raise ValueError("not a hidden prediction seal")
    if value.get("hidden_run_id") != hidden_run_id:
        raise ValueError("prediction seal targets a different hidden run")
    if value.get("source_plan_content_sha256") != source_plan_content_sha256:
        raise ValueError("prediction seal belongs to a different source plan")
    if value.get("target_endpoint_seen") is not False:
        raise ValueError("prediction seal reports hidden endpoint access")
    _validate_sha256(value.get("predicted_mgp_sha256"), field="predicted_mgp_sha256")
    declared = _validate_sha256(value.get("content_sha256"), field="content_sha256")
    content = dict(value)
    content.pop("content_sha256", None)
    if sha256_json(content) != declared:
        raise ValueError("prediction seal content hash mismatch")
    return value


def _validate_runtime_execution(
    path: str | Path,
    *,
    hidden_run_id: str,
    prediction_seal: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise TypeError("runtime execution manifest must be an object")
    if value.get("format") != "GENOME_HIDDEN_RUNTIME_EXECUTION" or value.get("version") != "0.1.0":
        raise ValueError("not a hidden runtime execution manifest")
    if value.get("hidden_run_id") != hidden_run_id:
        raise ValueError("runtime execution targets a different hidden run")
    if value.get("target_endpoint_seen") is not False:
        raise ValueError("runtime execution reports hidden endpoint access")
    if value.get("prediction_manifest_sha256") != prediction_seal.get("prediction_manifest_sha256"):
        raise ValueError("runtime execution does not match the sealed prediction")
    _validate_sha256(value.get("candidate_state_sha256"), field="candidate_state_sha256")
    declared = _validate_sha256(value.get("content_sha256"), field="content_sha256")
    content = dict(value)
    content.pop("content_sha256", None)
    if sha256_json(content) != declared:
        raise ValueError("runtime execution manifest content hash mismatch")
    return value


def materialize_source_plan(
    plan: RoundOneSourcePlan,
    *,
    output_root: str | Path,
    splits: Iterable[str] = ("training", "development", "hidden"),
    reveal_hidden: bool = False,
    prediction_seal: str | Path | None = None,
    runtime_execution: str | Path | None = None,
    max_workers: int = 8,
) -> dict[str, Any]:
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    root = ensure_dir(output_root)
    hidden_lives = [life for life in plan.lives if life.split == "hidden"]
    if len(hidden_lives) != 1:
        raise ValueError("source plan must contain exactly one hidden life")
    if reveal_hidden:
        if prediction_seal is None:
            raise ValueError("revealing the hidden endpoint requires a prediction seal")
        seal = _validate_prediction_seal(
            prediction_seal,
            hidden_run_id=hidden_lives[0].run_id,
            source_plan_content_sha256=plan.to_dict()["content_sha256"],
        )
        if runtime_execution is None:
            raise ValueError("revealing the hidden endpoint requires a runtime execution manifest")
        _validate_runtime_execution(
            runtime_execution,
            hidden_run_id=hidden_lives[0].run_id,
            prediction_seal=seal,
        )

    from huggingface_hub import hf_hub_download

    jobs = list(
        iter_materializable_checkpoints(
            plan,
            splits=splits,
            reveal_hidden=reveal_hidden,
        )
    )

    def download_checkpoint(
        item: tuple[LifeSourcePlan, CheckpointSource],
    ) -> dict[str, Any]:
        life, checkpoint = item
        destination = root / "raw" / life.run_id / checkpoint.branch
        ensure_dir(destination)
        path = Path(
            hf_hub_download(
                repo_id=life.repository,
                filename=checkpoint.weight.name,
                revision=checkpoint.commit,
                local_dir=destination,
            )
        )
        actual_size = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_size != checkpoint.weight.size:
            raise ValueError(
                f"download size mismatch for {life.run_id}@{checkpoint.branch}: "
                f"{actual_size} != {checkpoint.weight.size}"
            )
        if checkpoint.weight.sha256 is not None and actual_sha256 != checkpoint.weight.sha256:
            raise ValueError(f"download hash mismatch for {life.run_id}@{checkpoint.branch}")
        return {
            "run_id": life.run_id,
            "split": life.split,
            "step": checkpoint.step,
            "branch": checkpoint.branch,
            "commit": checkpoint.commit,
            "path": str(path.relative_to(root)),
            "bytes": actual_size,
            "sha256": actual_sha256,
        }

    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_checkpoint, job): job for job in jobs}
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: (record["run_id"], record["step"]))

    tokenizer_root = ensure_dir(root / "tokenizer")
    tokenizer_records = []
    for file in plan.tokenizer.files:
        path = Path(
            hf_hub_download(
                repo_id=plan.tokenizer.repository,
                filename=file.name,
                revision=plan.tokenizer.commit,
                local_dir=tokenizer_root,
            )
        )
        actual_sha256 = sha256_file(path)
        if path.stat().st_size != file.size or (
            file.sha256 is not None and actual_sha256 != file.sha256
        ):
            raise ValueError(f"tokenizer file mismatch: {file.name}")
        tokenizer_records.append(
            {
                "name": file.name,
                "path": str(path.relative_to(root)),
                "bytes": file.size,
                "sha256": actual_sha256,
            }
        )

    receipt = {
        "format": "GENOME_POLYPYTHIA_DOWNLOAD_RECEIPT",
        "version": "0.1.0",
        "source_plan_content_sha256": plan.to_dict()["content_sha256"],
        "reveal_hidden": reveal_hidden,
        "prediction_seal_sha256": (
            None if prediction_seal is None else sha256_file(prediction_seal)
        ),
        "runtime_execution_sha256": (
            None if runtime_execution is None else sha256_file(runtime_execution)
        ),
        "checkpoint_count": len(records),
        "checkpoint_bytes": sum(record["bytes"] for record in records),
        "checkpoints": records,
        "tokenizer": tokenizer_records,
    }
    receipt["content_sha256"] = sha256_json(receipt)
    receipt_name = "download-revealed.json" if reveal_hidden else "download-sealed.json"
    write_json(root / receipt_name, receipt, canonical=True)
    return receipt
