"""
ClinicSense – Tabular Q-Learning Agent (Optional Part 4)
The agent decides, one patient at a time, whether to accept an extra booking
or stop for the day.

State  : (free_slots, expected_remaining_no_shows_bucket)
Action : 0 = stop booking  |  1 = accept extra booking
Reward : revenue from attended appointments – idle/wait/overtime penalties
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
from collections import defaultdict

from src.config import (
    COST_IDLE_SLOT, COST_WAITING_MINUTE, COST_OVERTIME_MINUTE,
    SLOT_DURATION_MIN, SLOTS_PER_DAY, FIGURES_DIR, RANDOM_STATE,
)

os.makedirs(FIGURES_DIR, exist_ok=True)


# ── State discretisation ──────────────────────────────────────────────────────
MAX_EXTRA        = 10   # maximum extra bookings the agent can accept
ENS_BUCKETS      = 5    # bucket the expected no-shows into 5 groups
# The maximum realistic expected no-shows for a SLOTS_PER_DAY-slot clinic
# at a ~20% no-show rate: 20 * 0.20 = 4. We use 8 as a generous upper bound
# so that each bucket covers ~1.6 expected no-shows.
MAX_EXPECTED_NS  = 8.0


def discretise_state(free_slots: int, expected_noshows: float) -> tuple:
    """
    Maps continuous state to discrete (free_slots_bucket, expected_ns_bucket).
    free_slots      : clipped to [0, MAX_EXTRA]
    expected_noshows: normalised by MAX_EXPECTED_NS then scaled to [0, ENS_BUCKETS-1]
    """
    fs  = int(np.clip(free_slots, 0, MAX_EXTRA))
    # Normalise expected no-shows into ENS_BUCKETS equally-spaced buckets
    ens = int(np.clip(
        expected_noshows / MAX_EXPECTED_NS * ENS_BUCKETS,
        0,
        ENS_BUCKETS - 1
    ))
    return (fs, ens)


# ── Revenue and cost model ────────────────────────────────────────────────────
REVENUE_PER_ATTEND = 200   # unit revenue per patient who shows up


def compute_reward(n_showed: int, n_slots: int) -> float:
    """Reward = revenue_attended – penalties."""
    attended = min(n_showed, n_slots)
    idle     = max(0, n_slots - n_showed)
    overflow = max(0, n_showed - n_slots)

    revenue   = attended * REVENUE_PER_ATTEND
    idle_cost = idle    * COST_IDLE_SLOT
    wait_cost = overflow * SLOT_DURATION_MIN * COST_WAITING_MINUTE
    ot_cost   = overflow * SLOT_DURATION_MIN * COST_OVERTIME_MINUTE

    return float(revenue - idle_cost - wait_cost - ot_cost)


# ── Clinic Episode ────────────────────────────────────────────────────────────
class ClinicEnv:
    """
    Simple deterministic-schedule clinic environment.
    The agent receives patients sequentially and decides to accept or stop.
    """

    def __init__(self, patient_probs: np.ndarray,
                 base_slots: int = SLOTS_PER_DAY,
                 rng=None):
        self.patient_probs = patient_probs
        self.base_slots    = base_slots
        self.rng           = rng or np.random.default_rng(RANDOM_STATE)
        self.reset()

    def reset(self):
        self.booked_probs  = []
        self.patient_idx   = 0
        self.done          = False
        return self._get_state()

    def _get_state(self):
        n_extra  = max(0, len(self.booked_probs) - self.base_slots)
        # expected no-shows = sum of (1 - attendance_prob) for each booked patient
        exp_ns   = sum(1 - p for p in self.booked_probs)
        return discretise_state(
            free_slots       = max(0, self.base_slots - len(self.booked_probs)),
            expected_noshows = exp_ns   # raw count; discretise_state normalises it
        )

    def step(self, action: int):
        """
        action 0 : stop booking → episode ends
        action 1 : accept next patient → patient added to bookings
        """
        if self.done:
            raise RuntimeError("Episode already done. Call reset().")

        if action == 0 or self.patient_idx >= len(self.patient_probs):
            # end-of-day simulation
            n_slots  = self.base_slots
            show_arr = self.rng.random(len(self.booked_probs)) < np.array(self.booked_probs)
            n_showed = show_arr.sum()
            reward   = compute_reward(int(n_showed), n_slots)
            self.done = True
            return self._get_state(), reward, True

        # accept next patient
        self.booked_probs.append(self.patient_probs[self.patient_idx])
        self.patient_idx += 1
        return self._get_state(), 0.0, False  # no immediate reward


# ── Q-Learning Agent ──────────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self,
                 alpha:   float = 0.1,
                 gamma:   float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.999,
                 epsilon_min:   float = 0.05):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.q_table       = defaultdict(lambda: np.zeros(2))

    def choose_action(self, state: tuple) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = 0.0 if done else np.max(self.q_table[next_state])
        target    = reward + self.gamma * best_next
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_agent(daily_pools: list,
                n_episodes: int = 3000,
                base_slots: int = SLOTS_PER_DAY) -> tuple:
    """
    Train a Q-learning agent over n_episodes simulated clinic days.
    Returns (agent, episode_rewards).
    """
    rng   = np.random.default_rng(RANDOM_STATE)
    agent = QLearningAgent()
    episode_rewards = []

    for ep in range(n_episodes):
        pool = daily_pools[ep % len(daily_pools)]
        env  = ClinicEnv(patient_probs=pool, base_slots=base_slots, rng=rng)
        state = env.reset()
        total_reward = 0.0

        for _ in range(len(pool) + 1):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (ep + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-500:])
            print(f"  Episode {ep+1:>5}/{n_episodes}  "
                  f"Avg Reward (last 500): {avg:.1f}  "
                  f"ε={agent.epsilon:.3f}")

    return agent, episode_rewards


# ── Evaluate agent ────────────────────────────────────────────────────────────
def evaluate_agent(agent: QLearningAgent,
                   daily_pools: list,
                   n_eval: int = 1000,
                   base_slots: int = SLOTS_PER_DAY) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 1)
    results = []
    for ep in range(n_eval):
        pool  = daily_pools[ep % len(daily_pools)]
        env   = ClinicEnv(patient_probs=pool, base_slots=base_slots, rng=rng)
        state = env.reset()
        total_reward = 0.0
        for _ in range(len(pool) + 1):
            action = int(np.argmax(agent.q_table[state]))  # greedy
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        results.append({"episode": ep, "reward": total_reward,
                         "n_booked": len(env.booked_probs)})
    return pd.DataFrame(results)


# ── Plot training curve ───────────────────────────────────────────────────────
def plot_training_curve(episode_rewards: list, window: int = 100,
                         save: bool = True):
    rewards = np.array(episode_rewards)
    rolling = pd.Series(rewards).rolling(window).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode Reward")
    ax.plot(rolling, color="navy", linewidth=2, label=f"Rolling Mean ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Q-Learning Agent – Training Curve", fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "rl_training_curve.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
