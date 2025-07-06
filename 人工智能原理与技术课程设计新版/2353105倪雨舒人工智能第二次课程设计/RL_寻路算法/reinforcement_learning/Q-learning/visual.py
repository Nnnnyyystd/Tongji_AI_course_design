
import matplotlib.pyplot as plt
import numpy as np

def visualize_path_from_q_table(env, q_table, max_steps=100):
    """
    根据 Q 表策略可视化路径（按最大 Q 值依次前进），绿色高亮路径。
    """
    # 找到起点（值为 3）
    start = None
    for row in range(env.HEIGHT):
        for col in range(env.WIDTH):
            if env.state_map[row][col] == 3:
                start = [row, col]
                break
        if start:
            break

    if not start:
        print("❗ 未找到起点")
        return

    current_state = start
    visited = set()
    path = [current_state]
    steps = 0

    while steps < max_steps:
        steps += 1
        state_key = str(current_state)

        if tuple(current_state) in visited:
            print(f"🔁 循环路径终止 at {current_state}")
            break
        visited.add(tuple(current_state))

        if state_key not in q_table:
            print(f"❗ Q 表中无状态 {state_key}")
            break

        # 选取当前状态下 Q 值最大的动作
        q_values = q_table[state_key]
        best_action = np.argmax(q_values)

        # 根据动作更新下一状态
        next_state = current_state.copy()
        if best_action == 0 and next_state[0] > 0:
            next_state[0] -= 1
        elif best_action == 1 and next_state[0] < env.HEIGHT - 1:
            next_state[0] += 1
        elif best_action == 2 and next_state[1] > 0:
            next_state[1] -= 1
        elif best_action == 3 and next_state[1] < env.WIDTH - 1:
            next_state[1] += 1
        else:
            print("⚠️ 非法动作或越界终止")
            break

        path.append(next_state)

        # ✅ 画出绿色高亮路径
        x1 = next_state[1] * env.UNIT + 3
        y1 = next_state[0] * env.UNIT + 3
        x2 = x1 + env.UNIT - 6
        y2 = y1 + env.UNIT - 6
        rect = env.canvas.create_rectangle(x1, y1, x2, y2, fill='lime', outline='black', width=1)
        env.canvas.tag_raise(rect)

        # ✅ 到达终点则停止
        if env.state_map[next_state[0]][next_state[1]] == 2:
            print("🎯 到达终点")
            break

        current_state = next_state

    env.update()
    print(f"✅ 可视化策略路径完成，总步数：{len(path)}")






def visualize_path_from_statess(env, path):
    visited = set()
    for state in path:
        if tuple(state) in visited:
            continue
        visited.add(tuple(state))

        x1 = state[1] * env.UNIT + 3
        y1 = state[0] * env.UNIT + 3
        x2 = x1 + env.UNIT - 6
        y2 = y1 + env.UNIT - 6
        rect = env.canvas.create_rectangle(x1, y1, x2, y2, fill='lime', outline='black', width=1)
        env.canvas.tag_raise(rect)

    env.update()
    print(f"✅ 成功可视化最短路径，共 {len(path)} 步")

def plot_q_value_heatmap(env, q_table):

        print("heatmap =")
        #print(np.round(heatmap, 2))
        heatmap = np.full((env.HEIGHT, env.WIDTH), np.nan)  # NaN 表示不可达区域（障碍）

        for row in range(env.HEIGHT):
            for col in range(env.WIDTH):
                state = [row, col]
                key = str(state)
                if env.state_map[row][col] == 1:
                    continue  # 障碍物
                if key in q_table:
                    max_q = max(q_table[key])
                    heatmap[row, col] = max_q

        plt.figure(figsize=(8, 6))
        plt.title("🔥 Q-value Heatmap (Max per State)")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.imshow(heatmap, cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(label='Max Q Value')
        #plt.gca().invert_yaxis()  # 保持坐标系和环境一致（起点在上）
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()
    # ✅ 可视化 reward 趋势
def plot_rewards(reward_list):
        plt.figure(figsize=(10, 5))
        plt.plot(reward_list, label='Total Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Trend Over Training')
        plt.grid(True)
        plt.legend()
        plt.show()
def plot_training_statistics(all_total_episodes,all_first_success,all_shortest_found,all_min_steps):
    runs = list(range(1, len(all_total_episodes) + 1))

    plt.figure(figsize=(12, 6))
    plt.plot(runs, all_total_episodes, label="sum_episode", marker='o')
    plt.plot(runs, all_first_success, label="first_success_episode", marker='^')
    plt.plot(runs, all_shortest_found, label="find_shortest_path_episode", marker='s')
    plt.xlabel("episode_number")
    plt.ylabel("Episode")
    plt.title("critical_issue_contrast")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 最短路径步数分布（直方图）
    plt.figure(figsize=(8, 4))
    plt.hist(all_min_steps, bins=range(min(all_min_steps), max(all_min_steps)+1), edgecolor='black')
    plt.title("distribution_of_spath_step")
    plt.xlabel("shortest_path_step")
    plt.ylabel("appear_time")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    # 可选：打印统计信息
    print("平均训练轮数：", np.mean(all_total_episodes))
    print("平均首次成功轮数：", np.mean(all_first_success))
    print("平均最短路径发现轮数：", np.mean(all_shortest_found))
    print("平均最短路径步数：", np.mean(all_min_steps))