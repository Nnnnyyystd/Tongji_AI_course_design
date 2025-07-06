import matplotlib.pyplot as plt
import numpy as np
import random
from environment import Env
from collections import defaultdict
from visual import plot_rewards, plot_q_value_heatmap, visualize_path_from_statess,plot_training_statistics,visualize_path_from_q_table

class QLearningAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        #self.epsilon = 0.1
        self.epsilon =1  # 初始 ε（大的话鼓励探索）
        self.epsilon_min = 0.1  # 最小 ε（训练后期仍保留少量探索）
        self.epsilon_decay = 0.9995  # 衰减系数（或用线性衰减也可以）

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
                # 找到新的最大值，清空并记录新索引
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                # 与当前最大值相等，也加入候选
                max_index_list.append(index)
        # 从所有最大值中随机选一个（避免策略固定）
        return random.choice(max_index_list)


if __name__ == "__main__":
    cnt = 200  # 设置重复训练次数
    env = Env()  # 创建强化学习环境（图形界面迷宫）

    # 用于统计每轮训练过程的关键指标
    all_total_episodes = []  # 每轮训练所需的 episode 总数
    all_first_success = []  # 每轮首次成功到达终点的 episode 编号
    all_shortest_found = []  # 每轮首次找到最短路径的 episode 编号
    all_min_steps = []  # 每轮的最短路径步数
    # 初始化 Q-learning 智能体
    agent = QLearningAgent(actions=list(range(env.n_actions)))
    rewards = []  # 保存每个 episode 的总奖励
    min_success_steps = float('inf')  # 最短成功路径步数初始化为无穷大
    tolerance = 2  # 剪枝容忍度，超出当前最短路径 + tolerance 即剪枝
    pause_after_better_path = 50  # 每次发现更短路径后暂停剪枝的轮数
    pruning_paused_until = -1  # 当前剪枝暂停截止轮次
    skip_count = 0  # 连续被剪掉的次数
    skip_count_threshold = 20  # 连续被剪掉达到该值后，训练终止
    max_episodes = 20000  # 每轮训练最多迭代的 episode 数
    find_shortest_path_epi = 0  # 记录第一次找到最短路径的 episode 编号
    first_success_episode = None  # 首次成功到达终点的 episode 编号
    best_path = []  # 存储当前最短路径（状态序列）

    # episode 训练主循环
    for episode in range(max_episodes):
        state = env.reset()  # 重置环境，获得初始状态
        step_count = 0  # 当前 episode 步数计数器
        total_reward = 0  # 当前 episode 的总奖励
        success = False  # 是否成功到达终点
        episode_path = []  # 当前 episode 的状态路径
        # 判断是否启用剪枝（需首次成功到达终点 + 剪枝未暂停）
        pruning_enabled = (
                first_success_episode is not None and
                episode >= pruning_paused_until
        )

        # 单个 episode 的 step 循环
        while True:
            env.render()  # 可视化当前状态
            action = agent.get_action(str(state))  # 使用 ε-贪婪策略选动作
            episode_path.append(state)  # 记录路径
            next_state, reward, done = env.step(action)  # 执行动作，获取反馈
            step_count += 1
            total_reward += reward
            # ✅ 启动剪枝逻辑（成功之后 + 未暂停）
            if pruning_enabled and reward > 0 and step_count > min_success_steps + tolerance:
                #early_terminated = True
                print(f"[{episode}] ⚠️ 剪枝：路径 {step_count} 步 > 当前最优 {min_success_steps} 步")
                skip_count += 1
                break  # 当前路径明显比已有最短路径差 → 提前终止
            # Q-learning 公式更新 Q 值
            agent.learn(str(state), action, reward, str(next_state))
            state = next_state  # 状态更新
            if done:  # episode 结束（到达终点或撞墙）
                if reward > 0 :
                    success = True
                    # ✅ 第一次成功到达终点，记录 episode 编号
                    if first_success_episode is None:
                        first_success_episode = episode
                    # 如果找到更短路径，更新最短路径信息
                    if step_count < min_success_steps:
                        min_success_steps = step_count
                        skip_count = 0  # 剪枝命中次数清零
                        pruning_paused_until = episode + pause_after_better_path
                        find_shortest_path_epi = episode
                        best_path = episode_path.copy()
                        print(f"[{episode}] 🎯 发现更短路径：{step_count} 步 → 暂停剪枝至 Episode {pruning_paused_until}")
                break  # episode 结束
        # 在环境上打印 Q 表值（用于可视化）
        env.print_value_all(agent.q_table)
        rewards.append(total_reward)
        #  ε 衰减策略
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        #  剪枝命中次数超过阈值 → 判定训练收敛，停止
        if skip_count >= skip_count_threshold:
            print("🏁 训练完成！推测最短路径为：", min_success_steps)
            print("训练总轮次:", episode)
            print("首次成功抵达终点的轮次:", first_success_episode)
            print("首次找到最短路径的轮次:", find_shortest_path_epi)
            # 记录统计数据
            all_total_episodes.append(episode)
            all_first_success.append(first_success_episode)
            all_shortest_found.append(find_shortest_path_epi)
            all_min_steps.append(min_success_steps)
            break
        # 每隔 10 轮打印训练状态
        if episode % 10 == 0:
            print(f"[Episode {episode}] Steps: {step_count}, Reward: {total_reward}, Success: {success}")
    plot_q_value_heatmap(env, agent.q_table)
    plot_rewards(rewards)
    if 'best_path' in locals() and len(best_path) > 0:
        print("🎯 最终最短路径步数：", len(best_path))
        visualize_path_from_statess(env, best_path)
    else:
        print("⚠️ 未记录到成功路径，无法可视化")

    env.mainloop()
#   while(cnt>0):


   # plot_training_statistics(all_total_episodes, all_first_success, all_shortest_found, all_min_steps)
















