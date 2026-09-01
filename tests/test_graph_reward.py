"""Tests for Track A + Track B upgrades.

Run with:
    pytest tests/test_graph_reward.py -v
"""
import warnings
from unittest.mock import patch

import pytest
import torch

from graph_builder import build_graph
from reward import compute_reward
from solver import check_contradiction


# ---------------------------------------------------------------------------
# A2 — build_graph returns (graph, n_components)
# ---------------------------------------------------------------------------

def test_graph_order_and_reward():
    subproblems = [
        {"id": "q1", "text": "First", "type": "factual", "depends_on": []},
        {"id": "q2", "text": "Second", "type": "relational", "depends_on": ["q1"]},
    ]
    graph, n_components = build_graph(subproblems)
    assert list(graph.edges()) == [("q1", "q2")]
    assert n_components == 1
    reward = compute_reward({"q1": ["evidence"], "q2": []}, "answer", False)
    assert "score" in reward
    assert "components" in reward
    assert "explanation" in reward
    comps = reward["components"]
    assert set(comps.keys()) == {
        "correctness", "retrieval_presence", "token_efficiency",
        "logical_consistency", "grounding"
    }


def test_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        build_graph([
            {"id": "q1", "text": "A", "depends_on": ["q2"]},
            {"id": "q2", "text": "B", "depends_on": ["q1"]},
        ])


# ---------------------------------------------------------------------------
# A2 — disconnected-component warning
# ---------------------------------------------------------------------------

def test_disconnected_components_warning():
    subproblems = [
        {"id": "q1", "text": "Who is X?", "type": "factual", "depends_on": []},
        {"id": "q2", "text": "Who is Y?", "type": "factual", "depends_on": []},
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph, n_components = build_graph(subproblems)
    assert n_components == 2
    assert any("disconnected" in str(w.message).lower() for w in caught)


def test_connected_graph_no_warning():
    subproblems = [
        {"id": "q1", "text": "First", "type": "factual", "depends_on": []},
        {"id": "q2", "text": "Second", "type": "relational", "depends_on": ["q1"]},
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph, n_components = build_graph(subproblems)
    assert n_components == 1
    assert not any("disconnected" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# A4 — check_contradiction
# ---------------------------------------------------------------------------

def test_check_contradiction_detects_conflict():
    with patch("solver.call_llm", return_value="YES"):
        result = check_contradiction(
            "Paris is the capital of Germany.",
            {"q1": "Paris is the capital of France."},
        )
    assert result is True


def test_check_contradiction_no_conflict():
    with patch("solver.call_llm", return_value="NO"):
        result = check_contradiction(
            "The Eiffel Tower is in Paris.",
            {"q1": "Paris is the capital of France."},
        )
    assert result is False


def test_check_contradiction_empty_prior():
    result = check_contradiction("Some answer.", {})
    assert result is False


# ---------------------------------------------------------------------------
# A6 — four-component reward
# ---------------------------------------------------------------------------

def test_reward_all_supported_no_contradiction():
    reward = compute_reward(
        {"q1": [{"text": "The sky is blue."}], "q2": [{"text": "Water is wet."}]},
        "The sky is blue and water is wet.",
        contradictions_found=False,
        completion_tokens=100,
    )
    assert reward["components"]["retrieval_presence"] == 1.0
    assert reward["components"]["logical_consistency"] == 1.0
    assert reward["components"]["token_efficiency"] > 0.9
    assert 0.0 <= reward["components"]["grounding"] <= 1.0
    assert isinstance(reward["score"], float)


def test_reward_contradiction_penalty():
    reward_clean = compute_reward({"q1": ["e"]}, "answer", contradictions_found=False)
    reward_bad = compute_reward({"q1": ["e"]}, "answer", contradictions_found=True)
    assert reward_clean["score"] > reward_bad["score"]


def test_reward_empty_answer():
    reward = compute_reward({"q1": []}, "", contradictions_found=False)
    assert "empty" in reward["explanation"].lower()


# ---------------------------------------------------------------------------
# A7 — EM / F1 helpers
# ---------------------------------------------------------------------------

def test_exact_match():
    from eval import exact_match
    assert exact_match("London", "London") == 1.0
    assert exact_match("london", "LONDON") == 1.0
    assert exact_match("London", "Paris") == 0.0
    assert exact_match("the cat", "cat") == 1.0


def test_f1_score():
    from eval import f1_score
    assert f1_score("London is the capital", "London is the capital") == 1.0
    assert f1_score("London", "Paris") == 0.0
    f1 = f1_score("London is sunny", "London is cold")
    assert 0.0 < f1 < 1.0


# ---------------------------------------------------------------------------
# B1 — Policy network forward-pass shapes
# ---------------------------------------------------------------------------

def test_graph_edge_policy_shape():
    from policies import GraphEdgePolicy, build_obs_G
    pi = GraphEdgePolicy()
    obs = build_obs_G("factual", "relational", n_nodes=3, n_edges=2)
    assert obs.shape == (10,)
    out = pi(obs)
    assert out.action.shape == (1,)
    assert out.log_prob.shape == (1,)
    assert out.value.shape == (1,)
    assert out.action.item() in (0, 1)


def test_retrieval_select_policy_shape():
    from policies import RetrievalSelectPolicy, build_obs_R
    pi = RetrievalSelectPolicy()
    p_emb = torch.randn(384)
    q_emb = torch.randn(384)
    obs = build_obs_R(p_emb, q_emb)
    assert obs.shape == (768,)
    out = pi(obs)
    assert out.action.item() in (0, 1)


def test_context_keep_policy_shape():
    from policies import ContextKeepPolicy, build_obs_C
    pi = ContextKeepPolicy()
    p_emb = torch.randn(384)
    obs = build_obs_C(p_emb, ctx_tokens=512)
    assert obs.shape == (385,)
    out = pi(obs)
    assert out.action.item() in (0, 1)


def test_policy_evaluate_gradients():
    """evaluate() must return tensors that support backward for PPO updates."""
    from policies import GraphEdgePolicy, build_obs_G
    pi = GraphEdgePolicy()
    obs = build_obs_G("temporal", "comparative", n_nodes=2, n_edges=1)
    out = pi(obs)
    log_prob, value, entropy = pi.evaluate(obs.unsqueeze(0), out.action)
    loss = -log_prob.mean() + 0.5 * value.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in pi.parameters())


def test_type_onehot():
    from policies import type_onehot, SUBPROBLEM_TYPES
    for t in SUBPROBLEM_TYPES:
        vec = type_onehot(t)
        assert vec.shape == (4,)
        assert vec.sum().item() == 1.0
    # Unknown type maps to index 0
    vec = type_onehot("unknown_type")
    assert vec[0].item() == 1.0


# ---------------------------------------------------------------------------
# B2 — Rollout structure (no LLM / retriever needed)
# ---------------------------------------------------------------------------

def test_rollout_result_structure():
    from rollout import RolloutResult
    r = RolloutResult(
        question="test",
        final_answer="42",
        reward_score=0.1,
        reward_components={"retrieval_presence": 1.0},
    )
    assert r.question == "test"
    assert r.final_answer == "42"
    assert isinstance(r.traj_G, list)
    assert isinstance(r.traj_R, list)
    assert isinstance(r.traj_C, list)


def test_rollout_step_reward_assignment():
    from rollout import Step, _assign_reward
    steps = [
        Step(obs=torch.zeros(10), action=torch.tensor(1.0),
             log_prob=torch.tensor(-0.5), value=torch.tensor(0.2)),
        Step(obs=torch.zeros(10), action=torch.tensor(0.0),
             log_prob=torch.tensor(-0.7), value=torch.tensor(0.1)),
    ]
    _assign_reward(steps, 0.75)
    assert all(s.reward == 0.75 for s in steps)


# ---------------------------------------------------------------------------
# B3 — PPO helpers (no LLM needed)
# ---------------------------------------------------------------------------

def test_compute_gae_single_step():
    from train_ppo import compute_gae
    advantages, returns = compute_gae([1.0], [0.5], next_value=0.0)
    assert len(advantages) == 1
    assert abs(advantages[0] - 0.5) < 1e-5
    assert abs(returns[0] - 1.0) < 1e-5


def test_ppo_update_runs_without_error():
    from policies import GraphEdgePolicy
    from train_ppo import _ppo_update, _build_batch
    from rollout import Step

    pi = GraphEdgePolicy()
    opt = torch.optim.Adam(pi.parameters(), lr=1e-3)
    traj = []
    for _ in range(3):
        obs = torch.randn(10)
        out = pi(obs)
        traj.append(Step(
            obs=obs,
            action=out.action.squeeze(),
            log_prob=out.log_prob.squeeze(),
            value=out.value.squeeze(),
            reward=0.5,
        ))
    batch = _build_batch(traj)
    assert batch is not None
    losses = _ppo_update(pi, opt, batch, epochs=2)
    assert "policy_loss" in losses
    assert "value_loss" in losses
    assert "entropy" in losses


# ---------------------------------------------------------------------------
# B4 — Checkpoint save / load round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_save_load(tmp_path):
    from policies import (
        GraphEdgePolicy, RetrievalSelectPolicy, ContextKeepPolicy,
        save_checkpoint, load_latest_checkpoint,
    )
    import policies

    pi_G = GraphEdgePolicy()
    pi_R = RetrievalSelectPolicy()
    pi_C = ContextKeepPolicy()
    opt_G = torch.optim.Adam(pi_G.parameters(), lr=1e-3)
    opt_R = torch.optim.Adam(pi_R.parameters(), lr=1e-3)
    opt_C = torch.optim.Adam(pi_C.parameters(), lr=1e-3)
    history = [{"step": 0, "reward": 0.1}]

    original_dir = policies.CHECKPOINT_DIR
    policies.CHECKPOINT_DIR = tmp_path
    try:
        save_checkpoint(1, pi_G, pi_R, pi_C, opt_G, opt_R, opt_C, history)
        pi_G2 = GraphEdgePolicy()
        pi_R2 = RetrievalSelectPolicy()
        pi_C2 = ContextKeepPolicy()
        opt_G2 = torch.optim.Adam(pi_G2.parameters(), lr=1e-3)
        opt_R2 = torch.optim.Adam(pi_R2.parameters(), lr=1e-3)
        opt_C2 = torch.optim.Adam(pi_C2.parameters(), lr=1e-3)
        step, loaded = load_latest_checkpoint(pi_G2, pi_R2, pi_C2, opt_G2, opt_R2, opt_C2)
        assert step == 1
        assert loaded[0]["reward"] == 0.1
        orig_w = list(pi_G.parameters())[0]
        loaded_w = list(pi_G2.parameters())[0]
        assert torch.allclose(orig_w, loaded_w)
    finally:
        policies.CHECKPOINT_DIR = original_dir


# ---------------------------------------------------------------------------
# B-fix — 5th reward component: correctness (EM + F1 blend)
# ---------------------------------------------------------------------------

def test_reward_correctness_exact_match():
    """Perfect match → correctness = 1.0 (0.5*EM + 0.5*F1 = 1.0)."""
    reward = compute_reward(
        {"q1": [{"text": "London is the capital of England."}]},
        "London",
        contradictions_found=False,
        gold_answer="London",
    )
    assert reward["components"]["correctness"] == 1.0


def test_reward_correctness_partial_f1():
    """Partial overlap → 0 < correctness < 1."""
    reward = compute_reward(
        {"q1": [{"text": "Some evidence about cities."}]},
        "London is a big city",
        contradictions_found=False,
        gold_answer="London is a beautiful city",
    )
    cor = reward["components"]["correctness"]
    assert 0.0 < cor < 1.0


def test_reward_correctness_dominates_weight():
    """A correct answer should score higher than a wrong one, all else equal."""
    docs = {"q1": [{"text": "The sky is blue."}]}
    reward_correct = compute_reward(
        docs, "blue", contradictions_found=False, gold_answer="blue",
    )
    reward_wrong = compute_reward(
        docs, "red", contradictions_found=False, gold_answer="blue",
    )
    assert reward_correct["score"] > reward_wrong["score"]


def test_reward_no_gold_backward_compat():
    """No gold_answer → correctness defaults to 0.0 (backward compat)."""
    reward = compute_reward(
        {"q1": [{"text": "evidence"}]}, "answer", contradictions_found=False,
    )
    assert reward["components"]["correctness"] == 0.0
    assert "correctness" in reward["components"]
