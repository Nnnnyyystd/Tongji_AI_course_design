import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque


# 迷宫BFS搜索
def bfs_maze(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]#四个方向上下左右
    queue = deque([start]) #BFS需要队列数据结构
    visited = set([start]) #用于储存已经遍历过的点的集合
    parent = {start: None} #哈希表，用于储存各个节点的父节点
    frames = []  # 记录搜索过程的帧，便于后续的可视化
    while queue:
        x, y = queue.popleft()
        frames.append((x, y))  # 记录搜索路径
        if (x, y) == end:
            break
        for dx, dy in directions:
            nx, ny = x + dx, y + dy #遍历所有的相邻点
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited:
                queue.append((nx, ny)) #入队
                visited.add((nx, ny)) #标记
                parent[(nx, ny)] = (x, y) #记录父节点
    # 反向回溯最短路径
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path, frames


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
path, frames = bfs_maze(maze1, start, end)
print("最短路径:", path)

# 获取迷宫尺寸
rows, cols = len(maze1), len(maze1[0])

# 计算总帧数（搜索帧 + 最短路径帧）
total_frames = len(frames) + len(path)

# 动画绘制
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks(np.arange(-0.5, cols, 1))
ax.set_yticks(np.arange(-0.5, rows, 1))
ax.grid(True, color='black', linewidth=1)


def update(frame):
    ax.clear()
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))
    ax.grid(True, color='black', linewidth=0.5)
    ax.set_xlim([0, cols])
    ax.set_ylim([rows , 0])  # 反转y轴，使(0,0)在左上角
    # 画迷宫背景
    for x in range(rows):
        for y in range(cols):
            if maze1[x][y] == 1:
                ax.add_patch(plt.Rectangle((y, x), 1, 1, color='black', alpha=0.8))  # 墙壁
    # 画搜索过程
    if frame < len(frames):  # 只绘制 BFS 过程
        for i in range(frame + 1):
            x, y = frames[i]
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='blue', alpha=0.4))  # 搜索路径
    else:  # 画最短路径的动画
        for x, y in frames:  # 先显示 BFS 过的路径
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='blue', alpha=0.4))
        path_index = frame - len(frames)  # 计算路径索引
        for i in range(path_index + 1):  # 逐步将路径染红
            x, y = path[i]
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='red', alpha=0.6))
    # 画起点和终点
    ax.add_patch(plt.Rectangle((start[1], start[0]), 1, 1, color='green',  alpha=0.8))
    ax.add_patch(plt.Rectangle((end[1], end[0]), 1, 1, color='yellow',  alpha=0.8))

    ax.set_title(f"BFS Search: {frame + 1} / {total_frames}")


# 动画控制
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, repeat=False)
plt.show()
