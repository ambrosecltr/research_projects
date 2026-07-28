from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

from poetry50m.trajectory.types import SnapshotMetadata, WeightSnapshot


def make_snapshot(
    *,
    run_id: str = "r0",
    checkpoint_id: str,
    step: int,
    state_dict: Mapping[str, Tensor],
    initialization_id: str = "init-a",
    data_order_id: str = "order-a",
    architecture_signature: str = "decoder-v1",
    corpus_signature: str = "corpus-v1",
    model_config_hash: str = "model-hash",
    tokenizer_hash: str = "tokenizer-hash",
    code_signature: str = "code-hash",
    training_config_hash: str = "training-config-hash",
) -> WeightSnapshot:
    return WeightSnapshot(
        metadata=SnapshotMetadata(
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            step=step,
            initialization_id=initialization_id,
            data_order_id=data_order_id,
            architecture_signature=architecture_signature,
            corpus_signature=corpus_signature,
            model_config_hash=model_config_hash,
            tokenizer_hash=tokenizer_hash,
            code_signature=code_signature,
            training_config_hash=training_config_hash,
        ),
        state_dict={name: tensor.clone() for name, tensor in state_dict.items()},
    )
