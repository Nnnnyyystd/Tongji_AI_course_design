import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# A* 搜索算法
def astar_maze(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    open_set = []  # 优先队列
    heapq.heappush(open_set, (0, start))  # (f(n), 位置)

    g_score = {start: 0}  # 记录从起点到当前点的路径长度
    parent = {start: None}  # 记录路径

    frames = []  # 记录搜索过程，便于后续可视化

    def heuristic(a, b):
        """ 计算启发式函数 h(n)：曼哈顿距离 """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        frames.append(current)  # 记录搜索路径
        print("curr:",current)
        if current == end:
            break  # 找到终点

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor[0]][neighbor[1]] == 0:
                new_g = g_score[current] + 1  # 计算新 g(n)
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g  # 更新 g(n)
                    f_score = new_g + heuristic(neighbor, end)  # 计算 f(n)
                    heapq.heappush(open_set, (f_score, neighbor))
                    parent[neighbor] = current  # 记录父节点

    # 回溯路径
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path, frames


# 迷宫数据
maze1 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
start, end = (0, 0), (12, 14)

# 运行 A* 算法
path, frames = astar_maze(maze1, start, end)
print("最短路径:", path)

# 迷宫尺寸
rows, cols = len(maze1), len(maze1[0])

# 绘制动画
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
    ax.set_ylim([rows, 0])  # 反转 y 轴，使 (0,0) 在左上角

    # 绘制迷宫背景
    for x in range(rows):
        for y in range(cols):
            if maze1[x][y] == 1:
                ax.add_patch(plt.Rectangle((y, x), 1, 1, color='black', alpha=0.8))  # 墙壁

    # 绘制 A* 搜索路径
    if frame < len(frames):  # 先绘制 A* 搜索过程
        for i in range(frame + 1):
            x, y = frames[i]
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='blue', alpha=0.4))
    else:  # 绘制最短路径
        for x, y in frames:
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='blue', alpha=0.4))
        path_index = frame - len(frames)
        for i in range(path_index + 1):
            x, y = path[i]
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='red', alpha=0.6))

    # 画起点和终点
    ax.add_patch(plt.Rectangle((start[1], start[0]), 1, 1, color='green', alpha=0.8))
    ax.add_patch(plt.Rectangle((end[1], end[0]), 1, 1, color='yellow', alpha=0.8))

    ax.set_title(f"A* Search: {frame + 1} / {len(frames) + len(path)}")


# 运行动画
ani = animation.FuncAnimation(fig, update, frames=len(frames) + len(path), interval=50, repeat=False)
plt.show()
