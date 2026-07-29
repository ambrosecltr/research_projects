from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

from .hashing import sha256_file, sha256_json
from .io import atomic_write_json, load_json, load_yaml

SourceSplit = Literal["training", "development", "hidden"]


def pythia_repo(size: str, seed: int) -> str:
    return f"EleutherAI/pythia-{size}" if seed == 0 else f"EleutherAI/pythia-{size}-seed{seed}"


@dataclass(frozen=True)
class PythiaLifeSource:
    run_id: str
    size: str
    seed: int
    split: SourceSplit
    repository: str
    w0_revision: str = "step0"
    wt_revision: str = "step143000"
    w0_commit: str | None = None
    wt_commit: str | None = None
    licence: str = "Apache-2.0"
    dataset_repository: str = "EleutherAI/pile"
    order_repository: str = "EleutherAI/pile-preshuffled-seeds"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.repository != pythia_repo(self.size, self.seed):
            raise ValueError(f"repository mismatch for {self.size} seed{self.seed}")
        if self.split == "hidden" and self.wt_commit is not None:
            raise ValueError("hidden WT commit must remain unresolved before prediction sealing")

    @property
    def w0_ready(self) -> bool:
        return self.w0_commit is not None

    @property
    def wt_ready(self) -> bool:
        return self.split != "hidden" and self.wt_commit is not None


@dataclass(frozen=True)
class SourcePlan:
    lives: tuple[PythiaLifeSource, ...]
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_commit: str | None = None
    dataset_repository: str = "EleutherAI/pile"
    dataset_revision: str = "main"
    dataset_commit: str | None = None
    order_repository: str = "EleutherAI/pile-preshuffled-seeds"
    order_revision: str = "main"
    order_commit: str | None = None
    format: str = "GENOME_SOURCE_PLAN"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        ids = [life.run_id for life in self.lives]
        if len(ids) != len(set(ids)):
            raise ValueError("source-plan run IDs must be unique")
        assignments = {life.split for life in self.lives}
        if assignments != {"training", "development", "hidden"}:
            raise ValueError("source plan requires training, development and hidden lives")
        if any(life.size == "14m" and life.seed == 9 for life in self.lives):
            raise ValueError("the v1 source contract contains Pythia 14M seeds0–8 only")
        hidden = [life for life in self.lives if life.split == "hidden"]
        if len(hidden) != 1 or hidden[0].size != "31m" or hidden[0].seed != 9:
            raise ValueError("v1 requires exactly Pythia 31M seed9 as the fresh hidden life")

    @property
    def plan_id(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def pinned_for_materialization(self) -> bool:
        return bool(
            self.tokenizer_commit
            and all(life.w0_ready and (life.split == "hidden" or life.wt_ready) for life in self.lives)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "tokenizer_repository": self.tokenizer_repository,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_commit": self.tokenizer_commit,
            "dataset_repository": self.dataset_repository,
            "dataset_revision": self.dataset_revision,
            "dataset_commit": self.dataset_commit,
            "order_repository": self.order_repository,
            "order_revision": self.order_revision,
            "order_commit": self.order_commit,
            "lives": [asdict(life) for life in self.lives],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourcePlan":
        return cls(
            lives=tuple(PythiaLifeSource(**item) for item in value["lives"]),
            tokenizer_repository=str(value["tokenizer_repository"]),
            tokenizer_revision=str(value["tokenizer_revision"]),
            tokenizer_commit=value.get("tokenizer_commit"),
            dataset_repository=str(value.get("dataset_repository", "EleutherAI/pile")),
            dataset_revision=str(value.get("dataset_revision", "main")),
            dataset_commit=value.get("dataset_commit"),
            order_repository=str(value.get("order_repository", "EleutherAI/pile-preshuffled-seeds")),
            order_revision=str(value.get("order_revision", "main")),
            order_commit=value.get("order_commit"),
            format=str(value.get("format", "GENOME_SOURCE_PLAN")),
            version=str(value.get("version", "1.0.0")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SourcePlan":
        value = load_yaml(path) if str(path).endswith((".yaml", ".yml")) else load_json(path)
        return cls.from_dict(value)

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


def default_pythia_v1_plan() -> SourcePlan:
    lives: list[PythiaLifeSource] = []
    for size in ("14m", "31m"):
        for seed in range(10):
            if size == "14m" and seed == 9:
                continue
            split: SourceSplit = "training" if seed <= 7 else "development"
            if size == "31m" and seed == 9:
                split = "hidden"
            lives.append(
                PythiaLifeSource(
                    run_id=f"pythia-{size}-seed{seed}",
                    size=size,
                    seed=seed,
                    split=split,
                    repository=pythia_repo(size, seed),
                )
            )
    return SourcePlan(
        lives=tuple(lives),
        tokenizer_repository="EleutherAI/pythia-14m",
        tokenizer_revision="step0",
    )


def resolve_plan(plan: SourcePlan, *, token: str | None = None) -> SourcePlan:
    """Resolve requested refs to immutable commits without resolving the hidden WT."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("huggingface-hub is required to resolve source refs") from error
    api = HfApi(token=token)

    def model_sha(repository: str, revision: str) -> str:
        info = api.model_info(repository, revision=revision, files_metadata=False)
        if not info.sha:
            raise ValueError(f"Hugging Face did not return a commit for {repository}@{revision}")
        return str(info.sha)

    resolved_lives = []
    for life in plan.lives:
        resolved_lives.append(
            replace(
                life,
                w0_commit=model_sha(life.repository, life.w0_revision),
                wt_commit=None if life.split == "hidden" else model_sha(life.repository, life.wt_revision),
            )
        )
    dataset_commit = plan.dataset_commit
    order_commit = plan.order_commit
    try:
        info = api.dataset_info(plan.dataset_repository, revision=plan.dataset_revision)
        dataset_commit = None if not info.sha else str(info.sha)
    except Exception:
        dataset_commit = None
    try:
        info = api.dataset_info(plan.order_repository, revision=plan.order_revision)
        order_commit = None if not info.sha else str(info.sha)
    except Exception:
        order_commit = None
    return replace(
        plan,
        lives=tuple(resolved_lives),
        tokenizer_commit=model_sha(plan.tokenizer_repository, plan.tokenizer_revision),
        dataset_commit=dataset_commit,
        order_commit=order_commit,
    )


def materialize_plan(
    plan: SourcePlan,
    *,
    root: str | Path,
    token: str | None = None,
) -> dict[str, Any]:
    """Download W0 and allowed WT snapshots. Hidden WT is structurally impossible here."""
    if not plan.pinned_for_materialization:
        raise ValueError("source plan must be resolved and pinned before materialization")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("huggingface-hub is required for source materialization") from error
    root_path = Path(root)
    source_root = root_path / "source" / "hf"
    receipt_root = root_path / "source" / "receipts"
    source_root.mkdir(parents=True, exist_ok=True)
    receipt_root.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        "*.safetensors",
        "*.bin",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    receipts: list[dict[str, Any]] = []
    for life in plan.lives:
        revisions: list[tuple[str, str]] = [("w0", str(life.w0_commit))]
        if life.split != "hidden":
            revisions.append(("wt", str(life.wt_commit)))
        for role, commit in revisions:
            local_dir = source_root / life.run_id / role
            path = Path(
                snapshot_download(
                    repo_id=life.repository,
                    revision=commit,
                    cache_dir=str(root_path / "cache" / "huggingface"),
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    allow_patterns=allow_patterns,
                    token=token,
                )
            )
            files = [
                {
                    "path": str(file.relative_to(root_path)),
                    "bytes": file.stat().st_size,
                    "sha256": sha256_file(file),
                }
                for file in sorted(item for item in path.rglob("*") if item.is_file())
            ]
            receipt = {
                "run_id": life.run_id,
                "split": life.split,
                "role": role,
                "repository": life.repository,
                "commit": commit,
                "files": files,
            }
            atomic_write_json(receipt_root / f"{life.run_id}-{role}.json", receipt)
            receipts.append(receipt)
    summary = {"plan_id": plan.plan_id, "receipts": receipts}
    atomic_write_json(receipt_root / "materialization.json", summary)
    return summary


def reveal_hidden_endpoint(
    plan: SourcePlan,
    *,
    run_id: str,
    prediction_seal_path: str | Path,
    root: str | Path,
    token: str | None = None,
) -> dict[str, Any]:
    """Resolve and materialize hidden WT only after a valid prediction seal exists."""
    from .hidden import PredictionSeal

    seal = PredictionSeal.load(prediction_seal_path)
    if seal.run_id != run_id or seal.source_plan_id != plan.plan_id:
        raise ValueError("prediction seal does not match hidden life/source plan")
    hidden = next((life for life in plan.lives if life.run_id == run_id and life.split == "hidden"), None)
    if hidden is None:
        raise ValueError(f"{run_id!r} is not the hidden life")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("huggingface-hub is required to reveal hidden WT") from error
    api = HfApi(token=token)
    info = api.model_info(hidden.repository, revision=hidden.wt_revision, files_metadata=False)
    if not info.sha:
        raise ValueError("unable to resolve hidden endpoint commit")
    root_path = Path(root)
    destination = root_path / "source" / "revealed" / run_id / "wt"
    path = Path(
        snapshot_download(
            repo_id=hidden.repository,
            revision=str(info.sha),
            cache_dir=str(root_path / "cache" / "huggingface"),
            local_dir=str(destination),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.safetensors",
                "*.bin",
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
            token=token,
        )
    )
    files = [
        {
            "path": str(file.relative_to(root_path)),
            "bytes": file.stat().st_size,
            "sha256": sha256_file(file),
        }
        for file in sorted(item for item in path.rglob("*") if item.is_file())
    ]
    receipt = {
        "format": "GENOME_HIDDEN_REVEAL_RECEIPT",
        "version": "1.0.0",
        "run_id": run_id,
        "seal_id": seal.seal_id,
        "repository": hidden.repository,
        "commit": str(info.sha),
        "files": files,
    }
    receipt_path = root_path / "source" / "receipts" / f"{run_id}-wt-revealed.json"
    atomic_write_json(receipt_path, receipt)
    return receipt
