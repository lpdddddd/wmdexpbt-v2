"""Unit tests for EIR-MPC PBT elite broadcast (Phase D)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dexpbt_isaaclab.world_model.elite_recovery_bank import (
    EliteRecoveryBank,
    EliteRecoveryBankConfig,
)
from dexpbt_isaaclab.world_model.recovery_residual import (
    RecoveryResidualBridge,
    RecoveryResidualConfig,
    RecoveryResidualHead,
)
from dexpbt_isaaclab.world_model.eir_mpc import EIRBridge, EIRConfig
from dexpbt_isaaclab.world_model.elite_recovery_distill import ERDConfig
from dexpbt_isaaclab.world_model import eir_broadcast as bc
from dexpbt_isaaclab.world_model.eir_broadcast import (
    EliteBroadcastConfig,
    CONSUME_LIVE,
    CONSUME_RESTART,
)


_FEAT, _ACT = 4, 3


def _filled_bank(n=32, target=None, source_idx=0.0, nsucc_mean=40.0):
    cfg = EliteRecoveryBankConfig(
        enabled=True, bank_size=256, feature_dim=_FEAT, action_dim=_ACT,
        horizons=(4, 8), tol_threshold=0.02, hard_tol_threshold=0.012,
        near_success_mult=3.0, min_priority=0.0,
    )
    bank = EliteRecoveryBank(cfg)
    target = target if target is not None else torch.tensor([0.6, 0.6, 0.6])
    actions = (target.expand(n, _ACT) + 0.01 * torch.randn(n, _ACT)).clamp(-1, 1)
    d_hat = torch.full((n,), 0.015)
    tol = torch.full((n,), 0.015)
    lifted = torch.ones(n)
    targets = {
        "success_h4": torch.ones(n), "improve_h4": torch.zeros(n), "drop_h4": torch.zeros(n),
        "success_h8": torch.zeros(n), "improve_h8": torch.zeros(n), "drop_h8": torch.zeros(n),
    }
    import types
    cache = types.SimpleNamespace(
        actions_flat=actions, d_hat_flat=d_hat, success_tol_flat=tol,
        lifted_flat=lifted, targets_flat=targets, action_dim=_ACT,
    )
    feats = torch.randn(n, _FEAT)
    bank.update_from_cache(
        cache, feats,
        source_stats={"source_policy_idx": source_idx, "source_nsucc_mean": nsucc_mean},
    )
    return bank


def _bridge(bank=None):
    eir_cfg = EIRConfig(enabled=True, action_dim=_ACT, feature_dim=_FEAT)
    erd_cfg = ERDConfig(enabled=True, action_dim=_ACT, feature_dim=_FEAT, horizons=(4, 8))
    res_cfg = RecoveryResidualConfig(
        enabled=True, feature_dim=_FEAT, hidden_dim=8, action_dim=_ACT
    )
    res_bridge = RecoveryResidualBridge(cfg=res_cfg, head=RecoveryResidualHead(res_cfg))
    # a tiny scorer (uses ERD scorer architecture via attach is heavy; use a stub
    # nn.Module with state_dict for copy tests).
    scorer = torch.nn.Linear(_FEAT + _ACT, 4)
    return EIRBridge(
        eir_cfg=eir_cfg, erd_cfg=erd_cfg, scorer=scorer,
        residual_bridge=res_bridge, bank=bank,
        broadcast_cfg=EliteBroadcastConfig(enabled=True),
    )


class TestTrigger(unittest.TestCase):
    def test_should_broadcast_threshold(self):
        # high gap isolates the absolute-threshold path
        cfg = EliteBroadcastConfig(
            enabled=True, threshold_nsucc_mean=38.0, gap_over_pop_mean=1000.0,
            min_source_bank_size=1,
        )
        self.assertTrue(bc.should_broadcast(40.0, 10.0, cfg=cfg, source_bank_size=100))
        self.assertFalse(bc.should_broadcast(20.0, 10.0, cfg=cfg, source_bank_size=100))

    def test_should_broadcast_gap_over_pop(self):
        cfg = EliteBroadcastConfig(
            enabled=True, threshold_nsucc_mean=99.0, gap_over_pop_mean=6.0, min_source_bank_size=1
        )
        self.assertTrue(bc.should_broadcast(20.0, 12.0, cfg=cfg, source_bank_size=100))
        self.assertFalse(bc.should_broadcast(15.0, 12.0, cfg=cfg, source_bank_size=100))

    def test_should_broadcast_disabled_or_empty_bank(self):
        off = EliteBroadcastConfig(enabled=False)
        self.assertFalse(bc.should_broadcast(99.0, 0.0, cfg=off, source_bank_size=100))
        on = EliteBroadcastConfig(enabled=True, min_source_bank_size=100)
        self.assertFalse(bc.should_broadcast(99.0, 0.0, cfg=on, source_bank_size=10))

    def test_select_targets_bottom_k_excludes_source(self):
        cfg = EliteBroadcastConfig(enabled=True, bottom_k=2)
        pop = {0: 40.0, 1: 5.0, 2: 8.0, 3: 30.0, 4: 1.0}
        targets = bc.select_broadcast_targets(0, pop, cfg=cfg)
        self.assertEqual(targets, [4, 1])  # two weakest, source 0 excluded
        self.assertNotIn(0, targets)

    def test_decide_consume_mode(self):
        self.assertEqual(bc.decide_consume_mode(True), CONSUME_LIVE)
        self.assertEqual(bc.decide_consume_mode(False), CONSUME_RESTART)


class TestMailboxRoundtrip(unittest.TestCase):
    def test_write_read_merge_and_idempotent_consume(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d)
            src_bridge = _bridge(bank=_filled_bank(nsucc_mean=42.0))
            cfg = EliteBroadcastConfig(enabled=True, max_bank_merge_entries=1000)
            payload = bc.build_broadcast_payload(
                src_bridge, cfg=cfg, source_idx=0,
                source_nsucc_mean=42.0, source_nsucc_max=50.0, frame=1000,
            )
            self.assertGreater(payload["bank_entries"], 0)
            self.assertIn("residual_head", payload)
            self.assertIn("scorer", payload)

            bc.write_broadcast(ws, target_idx=3, payload=payload)
            meta = bc.read_broadcast(ws, 3)
            self.assertIsNotNone(meta)
            self.assertEqual(int(meta["source_policy"]), 0)

            # target with an empty bank merges the source entries
            empty_bank = EliteRecoveryBank(EliteRecoveryBankConfig(
                enabled=True, bank_size=256, feature_dim=_FEAT, action_dim=_ACT, horizons=(4, 8)
            ))
            tgt_bridge = _bridge(bank=empty_bank)
            self.assertEqual(tgt_bridge.bank.size, 0)
            loaded = bc.load_payload(meta)
            summary = bc.merge_payload_into_bridge(tgt_bridge, loaded, cfg=cfg)
            self.assertGreater(summary["merged_entries"], 0)
            self.assertTrue(summary["residual_copied"])
            self.assertTrue(summary["scorer_copied"])
            self.assertGreater(tgt_bridge.bank.size, 0)

            # mark consumed -> not pending anymore (idempotent)
            bc.mark_consumed(
                ws, 3, frame=1000, source_policy=0, mode=CONSUME_LIVE,
                merged_entries=summary["merged_entries"],
            )
            self.assertIsNone(bc.read_broadcast(ws, 3))

    def test_newer_frame_repending(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d)
            src = _bridge(bank=_filled_bank())
            cfg = EliteBroadcastConfig(enabled=True)
            p1 = bc.build_broadcast_payload(
                src, cfg=cfg, source_idx=1, source_nsucc_mean=40, source_nsucc_max=50, frame=100
            )
            bc.write_broadcast(ws, 2, p1)
            self.assertIsNotNone(bc.read_broadcast(ws, 2))
            bc.mark_consumed(ws, 2, frame=100, source_policy=1, mode=CONSUME_LIVE, merged_entries=5)
            self.assertIsNone(bc.read_broadcast(ws, 2))
            # a newer-frame payload becomes pending again
            p2 = bc.build_broadcast_payload(
                src, cfg=cfg, source_idx=1, source_nsucc_mean=41, source_nsucc_max=51, frame=200
            )
            bc.write_broadcast(ws, 2, p2)
            self.assertIsNotNone(bc.read_broadcast(ws, 2))

    def test_event_log_appended(self):
        with tempfile.TemporaryDirectory() as d:
            mdir = Path(d) / "metrics"
            bc.append_broadcast_event(mdir, {"event": "broadcast", "source": 0, "target": 3})
            bc.append_broadcast_event(mdir, {"event": "consume", "source": 0, "target": 3})
            lines = (mdir / "eir_broadcast_events.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
