from __future__ import annotations

from poetry50m.data.batch_stream import PreparedBatchStream
from poetry50m.data.packing import PackedSequence
from poetry50m.data.schema import ObjectiveMix
from poetry50m.training import TrainConfig
from poetry50m.workflows.exposure_plan import derived_train_config, plan_exposure


def test_exposure_plan_meets_the_token_target_and_records_actual_objective_exposure():
    packs = {
        "conditional_poetry": (
            PackedSequence(0, "conditional", ("conditional",), (1, 2, 3), (False, True, True)),
        ),
        "auxiliary_prose_ntp": (
            PackedSequence(
                0,
                "prose",
                ("prose",),
                (4, 5, 6, 7, 8),
                (False, True, True, True, True),
                "auxiliary_prose_ntp",
            ),
        ),
        "poetry_ntp": (
            PackedSequence(
                0,
                "verse",
                ("verse",),
                (9, 10, 11, 12),
                (False, True, True, True),
                "poetry_ntp",
            ),
        ),
    }
    plan = plan_exposure(
        PreparedBatchStream(
            packs,
            batch_size=1,
            pad_token_id=0,
            objective_mix=ObjectiveMix(0.1, 0.4, 0.5),
        ),
        parameter_count=10,
        tokens_per_parameter_per_pass=2,
        passes=2,
    )
    assert plan.target_data_tokens == 40
    assert plan.planned_data_tokens >= plan.target_data_tokens
    assert sum(plan.data_tokens_by_objective.values()) == plan.planned_data_tokens
    assert set(plan.data_tokens_by_objective) == {
        "conditional_poetry",
        "auxiliary_prose_ntp",
        "poetry_ntp",
    }


def test_derived_train_config_scales_a_valid_frozen_horizon():
    base = TrainConfig(
        max_steps=100,
        learning_rate=3e-4,
        warmup_steps=10,
        checkpoint_steps=(25, 50, 100),
        trajectory_capture_steps=(10, 100),
    )
    derived = derived_train_config(base, planned_steps=240)
    assert derived.max_steps == 240
    assert derived.warmup_steps == 24
    assert derived.checkpoint_steps == (60, 120, 240)
    assert derived.trajectory_capture_steps == (24, 240)
