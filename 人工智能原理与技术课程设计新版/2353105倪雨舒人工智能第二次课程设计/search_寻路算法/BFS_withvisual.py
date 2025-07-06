import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

def bfs_maze(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 四个方向
    queue = deque([start])  # BFS 队列
    visited = set([start])  # 记录访问过的点
    parent = {start: None}  # 记录路径
    frames = []  # 记录搜索帧
    order_map = {}  # 记录每个点入队的轮次

    round_counter = 0  # 轮次计数
    queue.append(None)  # 标记每轮的结束

    while queue:
        current = queue.popleft()
        if current is None:
            round_counter += 1
            if queue:
                queue.append(None)
            continue

        x, y = current
        frames.append((x, y))
        order_map[(x, y)] = round_counter  # 记录该点入队的轮次

        if (x, y) == end:
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited:
                queue.append((nx, ny))
                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)

    # 反向回溯最短路径
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()

    return path, frames, order_map  # 返回入队顺序信息

maze1 = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
start, end = (2, 3), (4, 7)
# 运行搜索
path, frames, order_map = bfs_maze(maze1, start, end)
rows, cols = len(maze1), len(maze1[0])
# 计算总帧数（搜索 + 最短路径）
total_frames = len(frames) + len(path)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks(np.arange(-0.5, cols, 1))
ax.set_yticks(np.arange(-0.5, rows, 1))
ax.grid(True, color='black', linewidth=1)
# 动画更新函数
def update(frame):
    ax.clear()
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))
    ax.grid(True, color='black', linewidth=0.5)
    ax.set_xlim([0, cols])
    ax.set_ylim([rows, 0])  # 反转y轴，使(0,0)在左上角

    # 画迷宫
    for x in range(rows):
        for y in range(cols):
            if maze1[x][y] == 1:
                ax.add_patch(plt.Rectangle((y, x), 1, 1, color='black', alpha=0.8))

    # 画搜索过程
    if frame < len(frames):
        for i in range(frame + 1):
            x, y = frames[i]
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='blue', alpha=0.4))
            ax.text(y + 0.5, x + 0.5, str(order_map[(x, y)]), fontsize=8, color='white', ha='center', va='center')

    else:  # 画最短路径
        for x, y in frames:
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='blue', alpha=0.4))
            ax.text(y + 0.5, x + 0.5, str(order_map[(x, y)]), fontsize=8, color='white', ha='center', va='center')

        path_index = frame - len(frames)
        for i in range(path_index + 1):
            x, y = path[i]
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='red', alpha=0.6))

    # 画起点和终点
    ax.add_patch(plt.Rectangle((start[1], start[0]), 1, 1, color='green', alpha=0.8))
    ax.add_patch(plt.Rectangle((end[1], end[0]), 1, 1, color='yellow', alpha=0.8))

    ax.set_title(f"BFS Search: {frame + 1} / {total_frames}")


# 运行动画
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, repeat=False)
plt.show()
