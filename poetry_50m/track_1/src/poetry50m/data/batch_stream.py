"""Checkpointable, objective-aware batches from canonical prepared packs."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
from typing import Any, Literal

import torch

from poetry50m.training.stream import Batch, SkippedBatchStats

from .artifacts import read_packed_sequences
from .packing import PackedSequence
from .schema import ObjectiveMix

CurriculumName = Literal["shuffled", "strict_hard_to_easy", "cyclic_hard_to_easy"]


class PreparedBatchStream:
    """Select whole objective batches by exact decimal weight, independent of group size."""

    def __init__(
        self,
        packs: Sequence[PackedSequence] | Mapping[str, Sequence[PackedSequence]],
        *,
        batch_size: int,
        pad_token_id: int,
        objective_mix: ObjectiveMix | None = None,
        curriculum: CurriculumName = "shuffled",
        seed: int = 0,
        difficulty: Mapping[str, float] | None = None,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
            or isinstance(pad_token_id, bool)
            or not isinstance(pad_token_id, int)
            or pad_token_id < 0
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise ValueError("batch_size, pad_token_id, and seed are invalid")
        if curriculum not in {"shuffled", "strict_hard_to_easy", "cyclic_hard_to_easy"}:
            raise ValueError(f"unsupported curriculum {curriculum}")
        objective_mix = objective_mix or ObjectiveMix()
        groups: dict[str, tuple[PackedSequence, ...]]
        if isinstance(packs, Mapping):
            groups = {name: tuple(items) for name, items in packs.items()}
        else:
            groups = {}
            for pack in packs:
                groups.setdefault(pack.objective, ())
                groups[pack.objective] += (pack,)
        weights = {
            "conditional_poetry": objective_mix.conditional_poetry,
            "auxiliary_prose_ntp": objective_mix.auxiliary_prose_ntp,
        }
        for name, weight in weights.items():
            if weight > 0.0 and not groups.get(name):
                raise ValueError(f"positive objective weight has no packs: {name}")
        self._groups = {
            name: values for name, values in groups.items() if weights.get(name, 0.0) > 0.0
        }
        if not self._groups:
            raise ValueError("no packs remain after applying objective weights")
        for name, values in self._groups.items():
            if not values or any(pack.objective != name for pack in values):
                raise ValueError("objective groups must be non-empty and match pack objective")
        self._batch_size = batch_size
        self._pad_token_id = pad_token_id
        self._curriculum = curriculum
        self._seed = seed
        self._difficulty = dict(difficulty or {})
        if any(
            not isinstance(row_id, str)
            or not row_id
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score < 0
            for row_id, score in self._difficulty.items()
        ):
            raise ValueError("difficulty scores must be finite non-negative numbers")
        if curriculum != "shuffled":
            missing = {
                self._row_id(pack)
                for values in self._groups.values()
                for pack in values
                if self._row_id(pack) not in self._difficulty
            }
            if missing:
                raise ValueError(f"difficulty is required for every pack row: {sorted(missing)!r}")
        fractions = {name: Fraction(str(weights[name])) for name in self._groups}
        denominator = math.lcm(*(fraction.denominator for fraction in fractions.values()))
        counts = {
            name: fraction.numerator * (denominator // fraction.denominator)
            for name, fraction in fractions.items()
        }
        divisor = math.gcd(*counts.values())
        self._schedule_counts = tuple(
            (name, counts[name] // divisor) for name in sorted(self._groups)
        )
        self._schedule_length = sum(count for _, count in self._schedule_counts)
        canonical = {
            name: [asdict(pack) for pack in values] for name, values in sorted(self._groups.items())
        }
        self._pack_hash = sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._stream_hash = sha256(
            json.dumps(
                {
                    "pack_hash": self._pack_hash,
                    "batch_size": batch_size,
                    "pad": pad_token_id,
                    "curriculum": curriculum,
                    "seed": seed,
                    "schedule_counts": self._schedule_counts,
                    "difficulty": sorted(self._difficulty.items()),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self._epochs = {name: 0 for name in self._groups}
        self._positions = {name: 0 for name in self._groups}
        self._orders = {name: self._order_for_epoch(name, 0) for name in self._groups}
        self._schedule_index = 0

    @classmethod
    def from_artifact(cls, packed_path: str, **kwargs: Any) -> PreparedBatchStream:
        from pathlib import Path

        return cls(read_packed_sequences(Path(packed_path)), **kwargs)

    def __iter__(self) -> PreparedBatchStream:
        return self

    @staticmethod
    def _row_id(pack: PackedSequence) -> str:
        return f"{pack.objective}:pack:{pack.pack_id}"

    def _pack_score(self, pack: PackedSequence) -> float:
        return self._difficulty[self._row_id(pack)]

    def _order_for_epoch(self, objective: str, epoch: int) -> tuple[int, ...]:
        values = self._groups[objective]
        indices = list(range(len(values)))
        if self._curriculum == "shuffled":
            random.Random(f"{self._seed}:{objective}:{epoch}").shuffle(indices)
            return tuple(indices)
        ranked = sorted(
            indices, key=lambda index: (-self._pack_score(values[index]), values[index].pack_id)
        )
        if self._curriculum == "cyclic_hard_to_easy":
            phase = epoch % len(ranked)
            ranked = ranked[phase:] + ranked[:phase]
        return tuple(ranked)

    @property
    def order_digest(self) -> str:
        payload = {
            "schedule_index": self._schedule_index,
            "epochs": self._epochs,
            "positions": self._positions,
            "orders": self._orders,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _next_packs(self) -> tuple[PackedSequence, ...]:
        offset = self._schedule_index
        objective: str | None = None
        for name, count in self._schedule_counts:
            if offset < count:
                objective = name
                break
            offset -= count
        if objective is None:
            raise AssertionError("objective schedule index is outside its compact cycle")
        self._schedule_index = (self._schedule_index + 1) % self._schedule_length
        order, position = self._orders[objective], self._positions[objective]
        end = min(position + self._batch_size, len(order))
        selected = tuple(self._groups[objective][index] for index in order[position:end])
        self._positions[objective] = end
        if end == len(order):
            self._epochs[objective] += 1
            self._positions[objective] = 0
            self._orders[objective] = self._order_for_epoch(objective, self._epochs[objective])
        return selected

    def _batch(self, packs: Sequence[PackedSequence]) -> Batch:
        rows, targets, losses = (
            [pack.input_ids[:-1] for pack in packs],
            [pack.input_ids[1:] for pack in packs],
            [pack.loss_mask[1:] for pack in packs],
        )
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), self._pad_token_id, dtype=torch.long)
        target_ids = torch.full((len(rows), width), self._pad_token_id, dtype=torch.long)
        loss_mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for index, (row, target, loss) in enumerate(zip(rows, targets, losses, strict=True)):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            target_ids[index, : len(target)] = torch.tensor(target, dtype=torch.long)
            loss_mask[index, : len(loss)] = torch.tensor(loss, dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "targets": target_ids,
            "loss_mask": loss_mask,
            "example_ids": tuple(self._row_id(pack) for pack in packs),
            "data_token_count": int(sum(len(row) for row in rows)),
        }

    def __next__(self) -> Batch:
        return self._batch(self._next_packs())

    def skip_batches(self, count: int) -> SkippedBatchStats:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("skip count must be a non-negative integer")
        data_tokens = sum(
            int(self._batch(self._next_packs())["data_token_count"]) for _ in range(count)
        )
        return SkippedBatchStats(batch_count=count, data_token_count=data_tokens)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 3,
            "pack_hash": self._pack_hash,
            "stream_hash": self._stream_hash,
            "epochs": dict(self._epochs),
            "positions": dict(self._positions),
            "schedule_index": self._schedule_index,
            "order_digest": self.order_digest,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        epochs, positions, schedule_index = (
            state.get("epochs"),
            state.get("positions"),
            state.get("schedule_index"),
        )
        if (
            state.get("pack_hash") != self._pack_hash
            or state.get("stream_hash") != self._stream_hash
            or state.get("format_version") != 3
            or not isinstance(epochs, dict)
            or not isinstance(positions, dict)
            or isinstance(schedule_index, bool)
            or not isinstance(schedule_index, int)
        ):
            raise ValueError("invalid prepared-stream state")
        if (
            set(epochs) != set(self._groups)
            or set(positions) != set(self._groups)
            or not 0 <= schedule_index < self._schedule_length
        ):
            raise ValueError("prepared-stream order mismatch")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in [*epochs.values(), *positions.values()]
        ):
            raise ValueError("prepared-stream order mismatch")
        self._epochs = dict(epochs)
        self._positions = dict(positions)
        self._orders = {
            name: self._order_for_epoch(name, self._epochs[name]) for name in self._groups
        }
        if any(self._positions[name] >= len(self._orders[name]) for name in self._groups):
            raise ValueError("prepared-stream order mismatch")
        self._schedule_index = schedule_index
        if state.get("order_digest") != self.order_digest:
            raise ValueError("prepared-stream order mismatch")
