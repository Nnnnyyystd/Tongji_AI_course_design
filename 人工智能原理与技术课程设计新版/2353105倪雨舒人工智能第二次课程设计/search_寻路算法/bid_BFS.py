import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# 双向 BFS 搜索
def bidirectional_bfs(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向

    queue1, queue2 = deque([start]), deque([end])  # BFS 队列
    visited1, visited2 = {start: 0}, {end: 0}  # 记录入队轮次
    parent1, parent2 = {start: None}, {end: None}  # 记录路径
    frames = []  # 记录搜索过程
    meeting_point = None  # 记录相遇点

    while queue1 and queue2:
        # 扩展起点方向的 BFS
        for _ in range(len(queue1)):
            x, y = queue1.popleft()
            frames.append((x, y, 1, visited1[(x, y)]))  # 标记 BFS1
            if (x, y) in visited2:  # 检测相遇
                meeting_point = (x, y)
                break
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited1:
                    queue1.append((nx, ny))
                    visited1[(nx, ny)] = visited1[(x, y)] + 1 #轮次递增
                    parent1[(nx, ny)] = (x, y)

        # 扩展终点方向的 BFS
        for _ in range(len(queue2)):
            x, y = queue2.popleft()
            frames.append((x, y, 2, visited2[(x, y)]))  # 标记 BFS2
            if (x, y) in visited1:  # 检测相遇
                meeting_point = (x, y)
                break
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited2:
                    queue2.append((nx, ny))
                    visited2[(nx, ny)] = visited2[(x, y)] + 1
                    parent2[(nx, ny)] = (x, y)

        if meeting_point:
            break  # 双向 BFS 相遇，停止搜索

    # 反向回溯路径
    path = []
    if meeting_point:
        cur = meeting_point
        while cur:
            path.append(cur)
            cur = parent1.get(cur)
        path.reverse()

        cur = parent2.get(meeting_point)
        while cur:
            path.append(cur)
            cur = parent2.get(cur)

    return path, frames, meeting_point

# 迷宫示例
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
path, frames, meeting_point = bidirectional_bfs(maze1, start, end)
print("最短路径:", path)
print("相遇点:", meeting_point)

# 计算总帧数
total_frames = len(frames) + len(path)

# 动画绘制
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks(np.arange(-0.5, len(maze1[0]), 1))
ax.set_yticks(np.arange(-0.5, len(maze1), 1))
ax.grid(True, color='black', linewidth=1)


def update(frame):
    ax.clear()
    ax.set_xticks(np.arange(0, len(maze1[0]), 1))
    ax.set_yticks(np.arange(0, len(maze1), 1))
    ax.grid(True, color='black', linewidth=0.5)
    ax.set_xlim([0, len(maze1[0])])
    ax.set_ylim([len(maze1), 0])  # 反转 y 轴

    # 画迷宫背景
    for x in range(len(maze1)):
        for y in range(len(maze1[0])):
            if maze1[x][y] == 1:
                ax.add_patch(plt.Rectangle((y, x), 1, 1, color='black', alpha=0.8))

    # 画搜索过程
    if frame < len(frames):
        for i in range(frame + 1):
            x, y, search_type, step = frames[i]
            color = "blue" if search_type == 1 else "orange"  # 起点搜索蓝色，终点搜索橙色
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color=color, alpha=0.4))
            ax.text(y + 0.3, x + 0.6, f"{step}", fontsize=8, color="black")  # 标注步数
    else:
        for x, y, search_type, step in frames:
            color = "blue" if search_type == 1 else "orange"
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color=color, alpha=0.4))
            ax.text(y + 0.3, x + 0.6, f"{step}", fontsize=8, color="black")

        path_index = frame - len(frames)
        for i in range(path_index + 1):
            x, y = path[i]
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='red', alpha=0.6))

    # 画起点、终点、相遇点
    ax.add_patch(plt.Rectangle((start[1], start[0]), 1, 1, color='green', alpha=0.8))
    ax.add_patch(plt.Rectangle((end[1], end[0]), 1, 1, color='yellow', alpha=0.8))
    if meeting_point:
        ax.add_patch(plt.Rectangle((meeting_point[1], meeting_point[0]), 1, 1, fill=False, edgecolor='lime', linewidth=2))

    ax.set_title(f"Bidirectional BFS: {frame + 1} / {total_frames}")


# 动画控制
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, repeat=False)
plt.show()
