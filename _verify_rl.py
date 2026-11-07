import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

print('=== RL Agent Integration Test ===')

from src.rl_agent import train_agent, evaluate_agent, QLearningAgent, ClinicEnv

rng = np.random.default_rng(42)
daily_pools = [rng.uniform(0.6, 0.95, 40) for _ in range(100)]

agent, rewards = train_agent(daily_pools, n_episodes=50, base_slots=20)
print('[1] Q-Learning training: OK  Q-table states:', len(agent.q_table))
print('    Epsilon:', round(agent.epsilon, 3))
print('    Mean reward last 20:', round(sum(rewards[-20:])/20, 1))

eval_df = evaluate_agent(agent, daily_pools, n_eval=30, base_slots=20)
mean_rew = round(float(eval_df['reward'].mean()), 1)
mean_bk  = round(float(eval_df['n_booked'].mean()), 1)
print('[2] Greedy evaluation: OK  mean_reward=' + str(mean_rew) + '  mean_bookings=' + str(mean_bk))

positive_days = int((eval_df['reward'] > 0).sum())
print('[3] Profitable days: ' + str(positive_days) + '/30')
print('[4] RL agent: ALL OK')
