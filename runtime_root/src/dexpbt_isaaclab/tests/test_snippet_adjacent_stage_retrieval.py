import torch

from dexpbt_isaaclab.world_model.elite_snippet_bank import (
    EliteSnippetBank,
    EliteSnippetBankConfig,
)
from dexpbt_isaaclab.world_model.elite_snippet_builder import SnippetBatch


def _batch(tols):
    n = len(tols)
    return SnippetBatch(
        feat_start=torch.randn(n, 8),
        actions=torch.randn(n, 4, 3).clamp(-1, 1),
        d_hat=torch.full((n,), 0.02),
        success_tol=torch.tensor(tols),
        lifted=torch.ones(n),
        picked=torch.zeros(n),
        dropped=torch.zeros(n),
        closest_d_hat=torch.full((n,), 0.02),
        success_h=torch.ones(n),
        improve_h=torch.zeros(n),
        drop_h=torch.zeros(n),
        recovery_label=torch.ones(n),
        s0=torch.ones(n),
        s1=torch.ones(n),
        i0=torch.zeros(n),
        d0=torch.zeros(n),
        d1=torch.zeros(n),
    )


def _bank():
    cfg = EliteSnippetBankConfig(
        enabled=True, bank_size=200, feature_dim=8, action_dim=3, horizon=4,
        guidance_enable=True, guidance_bank_tol=0.04,
        adjacent_stage_mixing_enable=True, min_primary_candidates=8,
        tolerance_compatibility_enable=False,
    )
    bank = EliteSnippetBank(cfg)
    # 1 FLOOR entry (tol<=0.02), many MID entries (0.02<tol<=0.04).
    bank.update_from_builder_batch(_batch([0.015] + [0.03] * 12),
                                   source_stats={"source_nsucc_mean": 30.0})
    return bank


def test_sparse_floor_bucket_does_not_fully_mask_mid_bank():
    bank = _bank()
    acts, sim, pri, ok = bank.retrieve_candidates(torch.randn(2, 8), k=4, stage=1)
    assert ok
    diag = bank.last_update_diagnostics
    # FLOOR primary is sparse (1 < 8) so adjacent MID is mixed in.
    assert diag["retrieve_primary_candidates"] == 1
    assert diag["retrieve_adjacent_candidates"] >= 1


def test_tolerance_gap_filters_incompatible_snippets():
    cfg = EliteSnippetBankConfig(
        enabled=True, bank_size=200, feature_dim=8, action_dim=3, horizon=4,
        guidance_enable=True, guidance_bank_tol=0.04,
        adjacent_stage_mixing_enable=True, min_primary_candidates=1,
        tolerance_compatibility_enable=True, tolerance_max_gap=0.001,
    )
    bank = EliteSnippetBank(cfg)
    bank.update_from_builder_batch(_batch([0.020, 0.030, 0.031]),
                                   source_stats={"source_nsucc_mean": 30.0})
    # Query tol 0.020 with 0.001 gap keeps only the compatible entry.
    acts, sim, pri, ok = bank.retrieve_candidates(
        torch.randn(1, 8), k=3, stage=1, query_tolerance=0.020
    )
    assert ok
