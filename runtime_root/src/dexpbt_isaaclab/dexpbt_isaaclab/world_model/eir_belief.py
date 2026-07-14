"""Belief-cache need predicate (Experiment C).

Kept free of train.py so unit tests call the same production helper.
"""

from __future__ import annotations


def need_eir_belief_cache(
    *,
    rssm_planner_present: bool,
    rssm_reranker_present: bool,
    snippet_belief_flag: bool,
    snippet_wm_rerank_enabled: bool,
    bridge_snippet_rerank_enabled: bool = False,
) -> bool:
    """Return True when the EIR RSSM belief cache must be built this rollout.

    Does not enable MPPI; it only decides whether belief encode is required for
    an already-enabled planner/reranker or for snippet WM rerank under the
    independent ``eir_enable_belief_for_snippet_wm_rerank`` flag.
    """
    if bool(rssm_planner_present) or bool(rssm_reranker_present):
        return True
    if bool(snippet_belief_flag) and (
        bool(snippet_wm_rerank_enabled) or bool(bridge_snippet_rerank_enabled)
    ):
        return True
    return False


def need_eir_belief_false_reason(
    *,
    rssm_planner_present: bool,
    rssm_reranker_present: bool,
    snippet_belief_flag: bool,
    snippet_wm_rerank_enabled: bool,
    bridge_snippet_rerank_enabled: bool = False,
) -> str:
    if need_eir_belief_cache(
        rssm_planner_present=rssm_planner_present,
        rssm_reranker_present=rssm_reranker_present,
        snippet_belief_flag=snippet_belief_flag,
        snippet_wm_rerank_enabled=snippet_wm_rerank_enabled,
        bridge_snippet_rerank_enabled=bridge_snippet_rerank_enabled,
    ):
        return ""
    if not snippet_belief_flag:
        return "snippet_belief_flag_off"
    if not (snippet_wm_rerank_enabled or bridge_snippet_rerank_enabled):
        return "snippet_wm_rerank_disabled"
    return "need_belief_false"
