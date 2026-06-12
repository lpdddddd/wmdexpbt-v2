"""Tests for the full RSSM ensemble (MPPI scoring only)."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from dexpbt_isaaclab.world_model.rssm_lite import RSSMLiteWorldModel, RSSMState
from dexpbt_isaaclab.world_model.rssm_ensemble import (
    RSSMEnsembleConfig,
    RSSMEnsembleManager,
    restore_rssm_ensemble_checkpoint,
)
from dexpbt_isaaclab.world_model.rssm_belief_provider import (
    build_rssm_belief_cache,
    BELIEF_OK,
)
from dexpbt_isaaclab.world_model.rssm_mppi_planner import (
    RSSMMPPIConfig,
    RSSMMPPIPlanner,
    ENSEMBLE_FB_ACTIVE,
    ENSEMBLE_FB_BELOW_MIN_VALID,
    ENSEMBLE_FB_NO_CACHE,
    ENSEMBLE_FB_PLANNER_INACTIVE,
)


T, N, OBS, ACT = 6, 4, 12, 4
A, D = 3, 6


def _real_rssm(*, aux_dim=0):
    return RSSMLiteWorldModel(
        obs_dim=OBS, action_dim=ACT, obs_embed_dim=32, action_embed_dim=16,
        h_dim=24, z_dim=8, head_hidden=32, aux_dim=aux_dim, aux_embed_dim=16,
    )


def _exp_buffer(*, with_task_aux=False):
    td = {
        "obses": torch.randn(T, N, OBS),
        "actions": torch.randn(T, N, ACT).clamp(-1, 1),
        "rewards": torch.randn(T, N, 1),
        "dones": torch.zeros(T, N, 1),
        "aux_d_hat": torch.rand(T, N, 1),
        "aux_failure": torch.zeros(T, N, 1),
        "aux_lifted": torch.ones(T, N, 1),
        "aux_success_cnt": torch.zeros(T, N, 1),
    }
    if with_task_aux:
        td.update({
            "aux_success_tolerance": torch.full((T, N, 1), 0.02),
            "aux_picked": torch.zeros(T, N, 1),
            "aux_dropped": torch.zeros(T, N, 1),
            "aux_closest_d_hat": torch.rand(T, N, 1),
        })
    return types.SimpleNamespace(tensor_dict=td)


def _trainer(model):
    return types.SimpleNamespace(model=model, device=torch.device("cpu"))


class _StubBeliefModel(nn.Module):
    """RSSM-like stub whose encode writes belief_seq[t,n,0]=n*T+t (+bias)."""

    belief_dim = 3
    aux_dim = 0

    def __init__(self, bias=0.0, fail=False):
        super().__init__()
        self.bias = float(bias)
        self.fail = bool(fail)
        self.heads = types.SimpleNamespace()

    def get_state_from_belief(self, b):
        return b

    def encode_sequence(self, batch, deterministic=False):
        if self.fail:
            raise RuntimeError("intentional encode failure")
        T_, N_ = batch.T, batch.B
        out = torch.zeros(T_, N_, self.belief_dim)
        for t in range(T_):
            for n in range(N_):
                out[t, n, 0] = float(n * T_ + t) + self.bias
        return out


class _ToyRSSM:
    """Toy RSSM for MPPI rollout scoring; success depends on action-target dist."""

    def __init__(self, target, base_unc=0.05):
        self.target = target
        self.base_unc = float(base_unc)

    def get_state_from_belief(self, belief):
        return RSSMState(belief=belief, deter=belief[:, :D // 2], stoch=belief[:, D // 2:])

    def imagine_step(self, state, action, *, deterministic=True, sample_temperature=1.0, **kwargs):
        if not isinstance(state, RSSMState):
            state = self.get_state_from_belief(state)
        belief = state.belief
        b = belief.shape[0]
        dev = belief.device
        d = (action - self.target.to(action.device)).norm(dim=-1)
        succ = torch.sigmoid(5.0 - 10.0 * d)
        drop = torch.sigmoid(-5.0 + 8.0 * d)
        preds = {
            "success": succ, "drop": drop, "recovery": succ.clone(),
            "value": -d, "task_error_delta": d * 0.1,
            "prior_std_mean": torch.full((b,), self.base_unc, device=dev),
            "imagine_mode": "prior_rollout",
        }
        nb = belief + 0.01 * action.mean(dim=-1, keepdim=True)
        return RSSMState(belief=nb, deter=nb[:, :D // 2], stoch=nb[:, D // 2:]), preds


def _cfg(**kw):
    base = dict(
        enabled=True, mode="full", start_after_frames=0, horizon=4,
        num_sequences=8, num_elite=4, cem_iters=1, uncertainty_threshold=10.0,
        min_sequence_margin=-1.0, include_elite_sequence=False,
    )
    base.update(kw)
    return RSSMMPPIConfig(**base)


# ---------------------------------------------------------------------------
# Ensemble manager
# ---------------------------------------------------------------------------
class TestEnsembleManager(unittest.TestCase):
    def test_rssm_ensemble_manager_constructs_independent_models(self):
        models = [_real_rssm() for _ in range(3)]
        mgr = RSSMEnsembleManager(models, RSSMEnsembleConfig(enabled=True, size=3))
        self.assertEqual(mgr.size, 3)
        self.assertEqual(len(mgr.models), 3)
        self.assertIsNot(mgr.models[0], mgr.models[1])
        # independent init -> at least one parameter differs across members
        p0 = next(mgr.models[0].parameters())
        p1 = next(mgr.models[1].parameters())
        self.assertFalse(torch.allclose(p0, p1))

    def test_rssm_ensemble_state_dict_roundtrip(self):
        models = [_real_rssm() for _ in range(3)]
        mgr = RSSMEnsembleManager(models, RSSMEnsembleConfig(enabled=True, size=3))
        state = mgr.state_dict()
        self.assertEqual(state["ensemble_size"], 3)
        self.assertEqual(len(state["members"]), 3)
        mgr2 = RSSMEnsembleManager([_real_rssm() for _ in range(3)], RSSMEnsembleConfig(size=3))
        loaded = mgr2.load_state_dict(state, strict=False)
        self.assertEqual(loaded, 3)
        # member 0 params now match
        p_a = next(mgr.models[0].parameters())
        p_b = next(mgr2.models[0].parameters())
        self.assertTrue(torch.allclose(p_a, p_b))

    def test_rssm_ensemble_eval_train_modes(self):
        models = [_real_rssm() for _ in range(2)]
        mgr = RSSMEnsembleManager(models, RSSMEnsembleConfig(size=2))
        mgr.eval()
        self.assertTrue(all(not m.training for m in mgr.models))
        mgr.train()
        self.assertTrue(all(m.training for m in mgr.models))


# ---------------------------------------------------------------------------
# Ensemble checkpoint / PBT restart persistence
# ---------------------------------------------------------------------------
class TestEnsembleCheckpointPersistence(unittest.TestCase):
    def _make_manager(self, size=3):
        models = [_real_rssm() for _ in range(size)]
        mgr = RSSMEnsembleManager(
            models, RSSMEnsembleConfig(enabled=True, size=size, load_mode="resume")
        )
        return mgr, models

    def test_rssm_ensemble_checkpoint_roundtrip(self):
        mgr, models = self._make_manager(3)
        for i, m in enumerate(models):
            with torch.no_grad():
                next(m.parameters()).add_(float(i + 1))

        payload = {
            "model_state_dict": models[0].state_dict(),
            "rssm_ensemble": mgr.state_dict(),
            "strategy": "rssm_belief_value",
        }
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "wm_seq_latest.pt"
            torch.save(payload, ckpt)
            loaded_payload = torch.load(ckpt, map_location="cpu", weights_only=False)

        mgr2, models2 = self._make_manager(3)
        metrics = restore_rssm_ensemble_checkpoint(
            mgr2,
            loaded_payload,
            wm_load_mode="auto",
            ensemble_load_mode="resume",
            primary_loaded=True,
        )
        self.assertEqual(metrics.get("ensemble_members_loaded"), 3.0)
        for i in range(3):
            p_a = next(mgr.models[i].parameters())
            p_b = next(mgr2.models[i].parameters())
            self.assertTrue(torch.allclose(p_a, p_b), msg=f"member {i}")

    def test_pbt_restart_restores_rssm_ensemble_members(self):
        mgr, models = self._make_manager(3)
        with torch.no_grad():
            next(models[1].parameters()).fill_(7.0)
            next(models[2].parameters()).fill_(9.0)

        payload = {
            "model_state_dict": models[0].state_dict(),
            "rssm_ensemble": mgr.state_dict(),
        }
        mgr_restart, restart_models = self._make_manager(3)
        metrics = restore_rssm_ensemble_checkpoint(
            mgr_restart,
            payload,
            wm_load_mode="auto",
            ensemble_load_mode="reset",
            primary_loaded=True,
        )
        self.assertEqual(metrics.get("ensemble_members_loaded"), 3.0)
        self.assertNotIn("init_from_primary", metrics)
        self.assertTrue(torch.allclose(
            next(restart_models[1].parameters()),
            next(models[1].parameters()),
        ))
        self.assertTrue(torch.allclose(
            next(restart_models[2].parameters()),
            next(models[2].parameters()),
        ))

    def test_missing_ensemble_checkpoint_copies_primary_to_members(self):
        primary = _real_rssm()
        with torch.no_grad():
            next(primary.parameters()).fill_(3.5)
        members = [_real_rssm() for _ in range(2)]
        mgr = RSSMEnsembleManager(
            [primary] + members,
            RSSMEnsembleConfig(enabled=True, size=3, load_mode="resume"),
        )
        before = [next(m.parameters()).clone() for m in members]

        payload = {"model_state_dict": primary.state_dict()}
        metrics = restore_rssm_ensemble_checkpoint(
            mgr,
            payload,
            wm_load_mode="auto",
            ensemble_load_mode="resume",
            primary_loaded=True,
        )
        self.assertEqual(metrics.get("init_from_primary"), 1.0)
        for i, m in enumerate(members, start=1):
            self.assertTrue(
                torch.allclose(next(m.parameters()), next(primary.parameters())),
                msg=f"member {i} should match primary",
            )
            self.assertFalse(torch.allclose(before[i - 1], next(m.parameters())))


# ---------------------------------------------------------------------------
# Belief cache ensemble
# ---------------------------------------------------------------------------
class TestBeliefCacheEnsemble(unittest.TestCase):
    def test_rssm_belief_cache_ensemble_valid(self):
        primary = _StubBeliefModel(bias=0.0)
        members = [primary, _StubBeliefModel(bias=100.0), _StubBeliefModel(bias=200.0)]
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(primary), experience_buffer=_exp_buffer(),
            returns_full=torch.randn(T, N), last_obs=torch.randn(N, OBS),
            last_d_hat=None, last_success_count=None,
            frames=0, generation=0, device=torch.device("cpu"),
            ensemble_models=members,
        )
        self.assertTrue(cache.valid)
        self.assertEqual(cache.fallback_reason, BELIEF_OK)
        self.assertIsNotNone(cache.belief_flat_ensemble)
        self.assertEqual(len(cache.belief_flat_ensemble), 3)
        self.assertEqual(cache.ensemble_size, 3)
        self.assertEqual(cache.ensemble_valid_size, 3)

    def test_rssm_belief_cache_slice_rows_ensemble(self):
        primary = _StubBeliefModel()
        members = [primary, _StubBeliefModel(bias=1.0), _StubBeliefModel(bias=2.0)]
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(primary), experience_buffer=_exp_buffer(),
            returns_full=torch.randn(T, N), last_obs=torch.randn(N, OBS),
            last_d_hat=None, last_success_count=None,
            frames=0, generation=0, device=torch.device("cpu"),
            ensemble_models=members,
        )
        sl = cache.slice_rows_ensemble(0, N)
        self.assertIsNotNone(sl)
        self.assertEqual(len(sl), 3)
        for b in sl:
            self.assertIsNotNone(b)
            self.assertEqual(int(b.shape[0]), N)
        # out of bounds -> entries None
        sl_bad = cache.slice_rows_ensemble(0, T * N + 1)
        self.assertTrue(all(x is None for x in sl_bad))

    def test_rssm_belief_cache_ensemble_partial_failure_fallback(self):
        primary = _StubBeliefModel()
        members = [primary, _StubBeliefModel(bias=1.0), _StubBeliefModel(fail=True)]
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(primary), experience_buffer=_exp_buffer(),
            returns_full=torch.randn(T, N), last_obs=torch.randn(N, OBS),
            last_d_hat=None, last_success_count=None,
            frames=0, generation=0, device=torch.device("cpu"),
            ensemble_models=members,
        )
        # primary cache still valid; ensemble has 2 valid + 1 failure.
        self.assertTrue(cache.valid)
        self.assertEqual(cache.ensemble_partial_failure_count, 1)
        self.assertEqual(cache.ensemble_valid_size, 2)
        self.assertIsNone(cache.belief_flat_ensemble[2])

    def test_rssm_belief_cache_with_aux_dim_12_ensemble(self):
        members = [_real_rssm(aux_dim=12) for _ in range(3)]
        cache = build_rssm_belief_cache(
            seq_wm_trainer=_trainer(members[0]),
            experience_buffer=_exp_buffer(with_task_aux=True),
            returns_full=torch.randn(T, N), last_obs=torch.randn(N, OBS),
            last_d_hat=torch.rand(N, 1), last_success_count=torch.zeros(N, 1),
            frames=0, generation=0, device=torch.device("cpu"),
            ensemble_models=members,
        )
        self.assertTrue(cache.valid)
        self.assertEqual(cache.ensemble_valid_size, 3)
        for b in cache.belief_flat_ensemble:
            self.assertEqual(int(b.shape[0]), T * N)
            self.assertEqual(int(b.shape[1]), members[0].belief_dim)


# ---------------------------------------------------------------------------
# MPPI ensemble scoring
# ---------------------------------------------------------------------------
class TestMPPIEnsemble(unittest.TestCase):
    def _inputs(self, b=2):
        belief = torch.randn(b, D)
        actor = torch.zeros(b, A)
        cand = torch.randn(b, 2, A).clamp(-1, 1)
        return belief, actor, cand

    def test_mppi_ensemble_scoring_shape(self):
        planner = RSSMMPPIPlanner(_cfg(use_model_ensemble=True, ensemble_min_valid_models=2))
        belief, actor, cand = self._inputs()
        models = [_ToyRSSM(torch.zeros(A)), _ToyRSSM(torch.full((A,), 0.3))]
        beliefs = [belief.clone(), belief.clone()]
        out = planner.plan(
            belief_flat=belief, actor_action=actor, candidate_actions=cand,
            rssm_model=models[0], frames=1000,
            belief_flat_ensemble=beliefs, rssm_models=models,
        )
        self.assertTrue(out["active"])
        self.assertEqual(planner.last_use_model_ensemble, 1.0)
        self.assertEqual(planner.last_ensemble_valid_size, 2.0)
        self.assertEqual(planner.last_ensemble_used, 1.0)
        self.assertEqual(planner.last_ensemble_configured, 1.0)
        self.assertEqual(planner.last_ensemble_available_size, 2.0)
        self.assertEqual(planner.last_ensemble_fallback_reason, float(ENSEMBLE_FB_ACTIVE))
        self.assertEqual(tuple(out["best_action"].shape), (2, A))

    def test_mppi_ensemble_imagine_mode_logged(self):
        planner = RSSMMPPIPlanner(_cfg(use_model_ensemble=True, ensemble_min_valid_models=2))
        belief, actor, cand = self._inputs()
        models = [_ToyRSSM(torch.zeros(A)), _ToyRSSM(torch.full((A,), 0.3))]
        beliefs = [belief.clone(), belief.clone()]
        planner.plan(
            belief_flat=belief, actor_action=actor, candidate_actions=cand,
            rssm_model=models[0], frames=1000,
            belief_flat_ensemble=beliefs, rssm_models=models,
        )
        self.assertIn(planner.last_imagine_mode, ("ensemble_prior_rollout", "ensemble_stochastic_prior_rollout"))

    def test_mppi_ensemble_falls_back_to_single_model_when_invalid(self):
        planner = RSSMMPPIPlanner(_cfg(use_model_ensemble=True, ensemble_min_valid_models=2))
        belief, actor, cand = self._inputs()
        models = [_ToyRSSM(torch.zeros(A)), _ToyRSSM(torch.full((A,), 0.3))]
        # only one valid belief -> below min_valid -> single model fallback
        beliefs = [belief.clone(), None]
        planner.plan(
            belief_flat=belief, actor_action=actor, candidate_actions=cand,
            rssm_model=models[0], frames=1000,
            belief_flat_ensemble=beliefs, rssm_models=models,
        )
        self.assertEqual(planner.last_use_model_ensemble, 0.0)
        self.assertEqual(planner.last_ensemble_used, 0.0)
        self.assertEqual(planner.last_ensemble_fallback_reason, float(ENSEMBLE_FB_BELOW_MIN_VALID))
        self.assertNotIn("ensemble", planner.last_imagine_mode)

    def test_mppi_ensemble_respects_min_valid_models(self):
        planner = RSSMMPPIPlanner(_cfg(use_model_ensemble=True, ensemble_min_valid_models=3))
        belief, actor, cand = self._inputs()
        models = [_ToyRSSM(torch.zeros(A)), _ToyRSSM(torch.full((A,), 0.3))]
        beliefs = [belief.clone(), belief.clone()]
        planner.plan(
            belief_flat=belief, actor_action=actor, candidate_actions=cand,
            rssm_model=models[0], frames=1000,
            belief_flat_ensemble=beliefs, rssm_models=models,
        )
        # 2 valid < min 3 -> fallback to single model
        self.assertEqual(planner.last_use_model_ensemble, 0.0)
        self.assertEqual(planner.last_ensemble_fallback_reason, float(ENSEMBLE_FB_BELOW_MIN_VALID))

    def test_mppi_ensemble_telemetry_no_cache(self):
        planner = RSSMMPPIPlanner(_cfg(use_model_ensemble=True, ensemble_min_valid_models=2))
        belief, actor, cand = self._inputs()
        models = [_ToyRSSM(torch.zeros(A)), _ToyRSSM(torch.full((A,), 0.3))]
        planner.plan(
            belief_flat=belief, actor_action=actor, candidate_actions=cand,
            rssm_model=models[0], frames=1000,
        )
        self.assertEqual(planner.last_ensemble_configured, 1.0)
        self.assertEqual(planner.last_ensemble_available_size, 0.0)
        self.assertEqual(planner.last_ensemble_used, 0.0)
        self.assertEqual(planner.last_ensemble_fallback_reason, float(ENSEMBLE_FB_NO_CACHE))

    def test_mppi_ensemble_planner_inactive_after_scoring(self):
        planner = RSSMMPPIPlanner(
            _cfg(
                use_model_ensemble=True,
                ensemble_min_valid_models=2,
                uncertainty_threshold=0.0,
                min_sequence_margin=1e9,
            )
        )
        belief, actor, cand = self._inputs()
        models = [_ToyRSSM(torch.zeros(A), base_unc=1.0), _ToyRSSM(torch.full((A,), 0.3), base_unc=1.0)]
        beliefs = [belief.clone(), belief.clone()]
        out = planner.plan(
            belief_flat=belief, actor_action=actor, candidate_actions=cand,
            rssm_model=models[0], frames=1000,
            belief_flat_ensemble=beliefs, rssm_models=models,
        )
        self.assertFalse(out["active"])
        self.assertEqual(planner.last_ensemble_used, 1.0)
        self.assertEqual(
            planner.last_ensemble_fallback_reason, float(ENSEMBLE_FB_PLANNER_INACTIVE)
        )

    def test_mppi_ensemble_no_double_risk_penalty(self):
        # With all ensemble weights AND single-model risk_beta at 0, the ensemble
        # score must equal the plain mean of per-model raw score_mean.
        planner = RSSMMPPIPlanner(
            _cfg(use_model_ensemble=True, ensemble_min_valid_models=2,
                 risk_beta=0.0, drop_cvar_weight=0.0, uncertainty_cvar_weight=0.0,
                 ensemble_risk_beta=0.0, ensemble_drop_weight=0.0,
                 ensemble_uncertainty_weight=0.0)
        )
        zeros = torch.zeros(1, 2)
        out_a = {"score_mean": torch.tensor([[3.0, 4.0]]), "score_std": zeros.clone(),
                 "drop_cvar": zeros.clone(), "uncertainty": zeros.clone(),
                 "score": torch.tensor([[3.0, 4.0]]), "imagine_mode": "prior_rollout"}
        out_b = {"score_mean": torch.tensor([[5.0, 6.0]]), "score_std": zeros.clone(),
                 "drop_cvar": zeros.clone(), "uncertainty": zeros.clone(),
                 "score": torch.tensor([[5.0, 6.0]]), "imagine_mode": "prior_rollout"}
        with mock.patch.object(planner, "_rollout_and_score", side_effect=[out_a, out_b]):
            res = planner._rollout_and_score_ensemble(
                belief_flat_list=[torch.zeros(1, D), torch.zeros(1, D)],
                seq=torch.zeros(1, 2, 4, A),
                actor_action=torch.zeros(1, A),
                rssm_models=[object(), object()],
            )
        self.assertTrue(torch.allclose(res["score"], torch.tensor([[4.0, 5.0]])))

    def test_mppi_ensemble_penalizes_disagreement(self):
        # seq0: members agree (std 0); seq1: same mean but high disagreement.
        planner = RSSMMPPIPlanner(
            _cfg(use_model_ensemble=True, ensemble_min_valid_models=2,
                 risk_beta=0.0, drop_cvar_weight=0.0, uncertainty_cvar_weight=0.0,
                 ensemble_risk_beta=1.0, ensemble_drop_weight=0.0,
                 ensemble_uncertainty_weight=0.0)
        )
        zeros = torch.zeros(1, 2)
        out_a = {"score_mean": torch.tensor([[10.0, 15.0]]), "score_std": zeros.clone(),
                 "drop_cvar": zeros.clone(), "uncertainty": zeros.clone(),
                 "score": zeros.clone(), "imagine_mode": "prior_rollout"}
        out_b = {"score_mean": torch.tensor([[10.0, 5.0]]), "score_std": zeros.clone(),
                 "drop_cvar": zeros.clone(), "uncertainty": zeros.clone(),
                 "score": zeros.clone(), "imagine_mode": "prior_rollout"}
        with mock.patch.object(planner, "_rollout_and_score", side_effect=[out_a, out_b]):
            res = planner._rollout_and_score_ensemble(
                belief_flat_list=[torch.zeros(1, D), torch.zeros(1, D)],
                seq=torch.zeros(1, 2, 4, A),
                actor_action=torch.zeros(1, A),
                rssm_models=[object(), object()],
            )
        # both raw means are 10, but seq1 has disagreement std=5 -> penalized.
        self.assertGreater(float(res["score"][0, 0]), float(res["score"][0, 1]))


if __name__ == "__main__":
    unittest.main()
