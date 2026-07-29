from __future__ import annotations

from pathlib import Path

from poetry50m.data.general_corpus import GeneralCorpusConfig
from poetry50m.data.general_sft import GeneralSftConfig, _training_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_v2_pretrain_config_is_exact_threefold_training_superset() -> None:
    v1 = GeneralCorpusConfig.load(
        PROJECT_ROOT / "configs/data/general_8m_pretrain.json"
    )
    v2 = GeneralCorpusConfig.load(
        PROJECT_ROOT / "configs/data/general_8m_pretrain_v2.json"
    )

    assert v2.train_rows == {
        source: count * 3 for source, count in v1.train_rows.items()
    }
    assert v2.heldout_rows == v1.heldout_rows
    assert v2.shard_order_seed == v1.shard_order_seed
    assert v2.split_salt == v1.split_salt
    assert sum(v2.train_rows.values()) * v2.context_length == 1_500_020_736


def test_v2_sft_config_uses_expanded_target_and_warmup() -> None:
    v1 = GeneralSftConfig.load(PROJECT_ROOT / "configs/data/general_8m_sft.json")
    v2 = GeneralSftConfig.load(
        PROJECT_ROOT / "configs/data/general_8m_sft_v2.json"
    )

    assert v2.selection_seed == v1.selection_seed
    assert v2.target_supervised_tokens == 60_000_000
    assert v2.warmup_steps == 90
    assert _training_config(v2, one_epoch_steps=2_220)["warmup_steps"] == 90
