import matplotlib.pyplot as plt
import numpy as np
import random
from environment import Env
from collections import defaultdict
from visual import plot_rewards, plot_q_value_heatmap, visualize_path_from_states

class QLearningAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        #self.epsilon = 0.1
        self.epsilon =0.1  # 初始 ε（鼓励探索）
        self.epsilon_min = 0.05  # 最小 ε（训练后期仍保留少量探索）
        self.epsilon_decay = 0.995  # 衰减系数（或用线性衰减也可以）

        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # 采样 <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # 贝尔曼方程更新
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

    # 从Q-table中选取动作
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 贪婪策略随机探索动作
            action = np.random.choice(self.actions)
        else:
            # 从q表中选择
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    rewards = []
    min_success_steps = float('inf')
    tolerance = 2
    pause_after_better_path = 100
    pruning_paused_until = -1
    skip_count = 0
    skip_count_threshold = 50
    max_episodes = 2000

    first_success_episode = None  # ✅ 首次成功路径出现时间
    best_path = []  # ✅ 最短路径状态序列（用于可视化）
    cnt=0
    for episode in range(max_episodes):
        state = env.reset()
        step_count = 0
        total_reward = 0
        success = False
        early_terminated = False
        episode_path = []

        # ✅ 是否启用剪枝
        pruning_enabled = (
                first_success_episode is not None and
                episode >= pruning_paused_until
        )

        while True:
            env.render()
            action = agent.get_action(str(state))
            episode_path.append(state)  # ✅ 路径记录
            next_state, reward, done = env.step(action)
            step_count += 1
            total_reward += reward

            # ✅ 剪枝：只在成功后剪 + 未暂停时启用
            #if pruning_enabled and reward > 0 and step_count > min_success_steps + tolerance:
             #   print(f"[{episode}] ⚠️ 剪枝：路径 {step_count} 步 > 当前最优 {min_success_steps} 步")
              #  early_terminated = True
               # skip_count += 1
               # break

            agent.learn(str(state), action, reward, str(next_state))
            state = next_state

            if done:
                if reward > 0 and not early_terminated:
                    cnt+=1
                    success = True

                    # ✅ 第一次成功，初始化剪枝启动时间
                    if first_success_episode is None:
                        first_success_episode = episode
                        pruning_paused_until = episode + 50  # 等 50 回合再剪

                    if step_count < min_success_steps:
                        min_success_steps = step_count
                        skip_count = 0
                        pruning_paused_until = episode + pause_after_better_path
                        best_path = episode_path.copy()
                        print(f"[{episode}] 🎯 发现更短路径：{step_count} 步 → 暂停剪枝至 Episode {pruning_paused_until}")
                break

        env.print_value_all(agent.q_table)
        rewards.append(total_reward)
        # ✅ ε 衰减
       # if agent.epsilon > agent.epsilon_min:
        #        agent.epsilon *= agent.epsilon_decay

        # ✅ 达到剪枝命中次数 → 判定收敛
        if skip_count >= skip_count_threshold:
            print("🏁 训练完成！推测最短路径为：", min_success_steps)
            break

        if episode % 10 == 0:
            print(f"[Episode {episode}] Steps: {step_count}, Reward: {total_reward}, Success: {success}")
            print(f"[Episode {episode}] 当前 ε = {agent.epsilon:.4f}")


    plot_q_value_heatmap(env, agent.q_table)
    plot_rewards(rewards)
    if 'best_path' in locals() and len(best_path) > 0:
        print("🎯 最终最短路径步数：", len(best_path))
        visualize_path_from_states(env, best_path)
    else:
        print("⚠️ 未记录到成功路径，无法可视化")

    env.mainloop()

