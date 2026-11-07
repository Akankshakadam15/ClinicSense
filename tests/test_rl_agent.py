"""Tests for src/rl_agent.py"""

import pytest
import numpy as np


# ── discretise_state ──────────────────────────────────────────────────────────
def test_discretise_state_zero():
    from src.rl_agent import discretise_state
    state = discretise_state(free_slots=0, expected_noshows=0.0)
    assert state == (0, 0)


def test_discretise_state_clipping():
    from src.rl_agent import discretise_state, MAX_EXTRA, ENS_BUCKETS
    # Values above max should be clipped
    state = discretise_state(free_slots=100, expected_noshows=9999.0)
    assert state[0] == MAX_EXTRA
    assert state[1] == ENS_BUCKETS - 1


def test_discretise_state_typical():
    """Typical clinic: 3 free slots, ~4 expected no-shows should give non-zero bucket."""
    from src.rl_agent import discretise_state, ENS_BUCKETS
    state = discretise_state(free_slots=3, expected_noshows=4.0)
    fs, ens = state
    assert 0 <= fs <= 3
    assert 0 <= ens < ENS_BUCKETS
    # With 4 expected no-shows out of MAX_EXPECTED_NS=8 → bucket 2 or 3
    assert ens >= 2, f"Expected bucket ≥2 for 4 no-shows, got {ens}"


def test_different_expected_noshows_give_different_buckets():
    """The fix ensures that different expected no-show values give different state buckets."""
    from src.rl_agent import discretise_state
    s_low  = discretise_state(free_slots=5, expected_noshows=1.0)
    s_high = discretise_state(free_slots=5, expected_noshows=7.0)
    assert s_low[1] != s_high[1], \
        "Low and high expected no-shows should map to different buckets after the bug fix"


# ── compute_reward ─────────────────────────────────────────────────────────────
def test_compute_reward_perfect_attendance():
    from src.rl_agent import compute_reward, REVENUE_PER_ATTEND
    from src.config import SLOTS_PER_DAY
    # All slots filled exactly → maximum revenue, no penalties
    reward = compute_reward(n_showed=SLOTS_PER_DAY, n_slots=SLOTS_PER_DAY)
    assert reward == SLOTS_PER_DAY * REVENUE_PER_ATTEND


def test_compute_reward_all_noshows():
    from src.rl_agent import compute_reward
    from src.config import SLOTS_PER_DAY, COST_IDLE_SLOT
    reward = compute_reward(n_showed=0, n_slots=SLOTS_PER_DAY)
    expected = -(SLOTS_PER_DAY * COST_IDLE_SLOT)
    assert reward == expected


def test_compute_reward_overflow():
    from src.rl_agent import compute_reward, REVENUE_PER_ATTEND
    from src.config import COST_WAITING_MINUTE, COST_OVERTIME_MINUTE, SLOT_DURATION_MIN
    n_slots = 10
    n_showed = 12   # 2 overflow patients
    reward = compute_reward(n_showed=n_showed, n_slots=n_slots)
    overflow = 2
    expected_revenue  = n_slots * REVENUE_PER_ATTEND
    expected_wait_cost = overflow * SLOT_DURATION_MIN * COST_WAITING_MINUTE
    expected_ot_cost   = overflow * SLOT_DURATION_MIN * COST_OVERTIME_MINUTE
    expected = expected_revenue - expected_wait_cost - expected_ot_cost
    assert reward == expected


# ── ClinicEnv ──────────────────────────────────────────────────────────────────
def test_clinic_env_reset():
    from src.rl_agent import ClinicEnv
    probs = np.full(30, 0.8)
    env = ClinicEnv(patient_probs=probs, base_slots=20)
    state = env.reset()
    assert isinstance(state, tuple)
    assert len(state) == 2


def test_clinic_env_stop_action():
    from src.rl_agent import ClinicEnv
    probs = np.full(30, 0.8)
    env = ClinicEnv(patient_probs=probs, base_slots=20)
    env.reset()
    # Accept a few patients then stop
    for _ in range(5):
        state, reward, done = env.step(1)
        if done:
            break
    state, reward, done = env.step(0)  # stop
    assert done is True
    assert isinstance(reward, float)


def test_clinic_env_accept_action():
    from src.rl_agent import ClinicEnv
    probs = np.full(30, 0.8)
    env = ClinicEnv(patient_probs=probs, base_slots=20)
    env.reset()
    state, reward, done = env.step(1)  # accept one patient
    assert done is False
    assert reward == 0.0  # no intermediate reward
    assert len(env.booked_probs) == 1


def test_clinic_env_done_raises():
    from src.rl_agent import ClinicEnv
    probs = np.full(5, 0.8)
    env = ClinicEnv(patient_probs=probs, base_slots=20)
    env.reset()
    env.step(0)  # stop immediately
    with pytest.raises(RuntimeError):
        env.step(1)


# ── QLearningAgent ─────────────────────────────────────────────────────────────
def test_agent_choose_action_greedy():
    from src.rl_agent import QLearningAgent
    agent = QLearningAgent(epsilon=0.0)  # fully greedy
    state = (5, 2)
    # Manually set Q values so action 1 is better
    agent.q_table[state][0] = 1.0
    agent.q_table[state][1] = 5.0
    assert agent.choose_action(state) == 1


def test_agent_update_changes_q():
    from src.rl_agent import QLearningAgent
    agent = QLearningAgent(alpha=1.0, gamma=0.0, epsilon=0.0)
    state, next_state = (5, 2), (4, 2)
    agent.q_table[state] = np.zeros(2)
    agent.update(state, action=1, reward=10.0, next_state=next_state, done=True)
    # With alpha=1, gamma=0, done=True → Q[state][1] should become reward
    assert agent.q_table[state][1] == pytest.approx(10.0)


def test_agent_epsilon_decays():
    from src.rl_agent import QLearningAgent
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.1)
    for _ in range(30):
        agent.decay_epsilon()
    assert agent.epsilon < 1.0
    assert agent.epsilon >= 0.1


# ── train_agent (small run) ───────────────────────────────────────────────────
def test_train_agent_runs(daily_pools):
    from src.rl_agent import train_agent
    agent, rewards = train_agent(daily_pools, n_episodes=20, base_slots=20)
    assert len(rewards) == 20
    assert len(agent.q_table) > 0


def test_evaluate_agent_returns_dataframe(daily_pools):
    from src.rl_agent import train_agent, evaluate_agent
    agent, _ = train_agent(daily_pools, n_episodes=10, base_slots=20)
    df = evaluate_agent(agent, daily_pools, n_eval=10, base_slots=20)
    assert "reward" in df.columns
    assert "n_booked" in df.columns
    assert len(df) == 10
