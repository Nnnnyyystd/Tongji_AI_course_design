import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from collections import defaultdict

np.random.seed(1) #随机种子
PhotoImage = ImageTk.PhotoImage #用于显示图片
UNIT = 60
HEIGHT = 13
WIDTH = 16


class Env(tk.Tk):  # 继承自 Tkinter 的主窗口类
    def __init__(self):
        super(Env, self).__init__()  # 初始化 Tk 父类
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.UNIT = UNIT
        self.action_space = ['u', 'd', 'l', 'r']# 定义动作空间：上、下、左、右
        self.n_actions = len(self.action_space)  # 动作数量为 4
        # 设置窗口标题与大小
        self.title('Q Learning')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))
        # 加载图像（智能体、终点、障碍物）
        self.shapes = self.load_images()
        # 构建网格画布
        self.canvas = self._build_canvas()
        # 存放所有 Q 值文本元素的列表，用于后续删除
        self.texts = []
        # 地图结构：0 可通行、1 障碍物、2 终点、3 起点
        self.state_map = [
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
]
        # 放置起点/终点/障碍物图形
        self._place_objects()
        # 记录某个格子被访问次数，用于“回头惩罚”
        self.visit_count = defaultdict(int)

    #绘制网格
    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white', height=HEIGHT * UNIT, width=WIDTH * UNIT)
        for c in range(0, WIDTH * UNIT, UNIT):
            canvas.create_line(c, 0, c, HEIGHT * UNIT)
        for r in range(0, HEIGHT * UNIT, UNIT):
            canvas.create_line(0, r, WIDTH * UNIT, r)
        canvas.pack()
        return canvas
    #放置元素
    def _place_objects(self):
        for row in range(HEIGHT):
            for col in range(WIDTH):
                x = col * self.UNIT + self.UNIT // 2  # ✅ 横向中心
                y = row * self.UNIT + self.UNIT // 2  # ✅ 纵向中心
                if self.state_map[row][col] == 3:
                    self.rectangle = self.canvas.create_image(x, y, image=self.shapes[0])
                elif self.state_map[row][col] == 2:
                    self.circle = self.canvas.create_image(x, y, image=self.shapes[2])
                elif self.state_map[row][col] == 1:
                    self.canvas.create_image(x, y, image=self.shapes[1])

    # 加载图片
    def load_images(self):
        rectangle = PhotoImage(Image.open("../img/rectangle.png").resize((25, 25)))
        triangle = PhotoImage(Image.open("../img/triangle.png").resize((25, 25)))
        circle = PhotoImage(Image.open("../img/circle.png").resize((25, 25)))
        return rectangle, triangle, circle
    # 绘制某个格子各个方向的Q值
    def text_value(self, row, col, contents, action, font='Helvetica', size=8,
                   style='normal', anchor="center"):
        center_x = col * self.UNIT + self.UNIT // 2
        center_y = row * self.UNIT + self.UNIT // 2
        offset = self.UNIT // 3  # 控制偏移量，约占格子边长的 1/3
        if action == 0:  # 上
            x, y = center_x, center_y - offset
        elif action == 1:  # 下
            x, y = center_x, center_y + offset
        elif action == 2:  # 左
            x, y = center_x - offset, center_y
        else:  # 右
            x, y = center_x + offset, center_y

        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents, font=font, anchor="nw")
        self.texts.append(text)
    # 显示Q值
    def print_value_all(self, q_table):
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for i in range(HEIGHT):
            for j in range(WIDTH):
                for action in range(4):
                    state = [i, j]
                    if str(state) in q_table.keys():
                        temp = q_table[str(state)][action]
                        self.text_value(i, j, round(temp, 2), action)
    # 像素坐标映射到状态二维数组中
    def coords_to_state(self, coords):
        col = int((coords[0] - self.UNIT // 2) / self.UNIT)
        row = int((coords[1] - self.UNIT // 2) / self.UNIT)
        return [row, col]

    # 格子位置的状态坐标映射到像素坐标
    def state_to_coords(self, state):
        row, col = state
        x = col * self.UNIT + self.UNIT // 2
        y = row * self.UNIT + self.UNIT // 2
        return [x, y]

    # 地图状态刷新
    def reset(self):
        self.update()
        time.sleep(0.1)
        self.visited = set()
        self.visit_count.clear()
        for row in range(HEIGHT):
            for col in range(WIDTH):
                if self.state_map[row][col] == 3:
                    x, y = col * UNIT + 50, row * UNIT + 50
                    self.canvas.coords(self.rectangle, x, y)
                    self.render()
                    return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        # 获取当前 Agent 的像素坐标位置
        state_coords = self.canvas.coords(self.rectangle)
        self.render()  # 可视化更新
        # 将像素坐标转换为网格状态坐标 [row, col]
        state = self.coords_to_state(state_coords)
        next_state = state.copy()
        # 根据动作更新状态
        if action == 0 and state[0] > 0:  # 向上
            next_state[0] -= 1
        elif action == 1 and state[0] < HEIGHT - 1:  # 向下
            next_state[0] += 1
        elif action == 2 and state[1] > 0:  # 向左
            next_state[1] -= 1
        elif action == 3 and state[1] < WIDTH - 1:  # 向右
            next_state[1] += 1
        # 如果尝试无效动作（撞墙或原地），给予惩罚并终止该步
        if next_state == state:
            return state, -10, False
        # 将图像对象移动到新位置
        target_coords = self.state_to_coords(next_state)
        self.canvas.coords(self.rectangle, target_coords[0], target_coords[1])
        self.canvas.tag_raise(self.rectangle)  # 将智能体图像置于最上层
        # 获取终点的位置，并计算当前与之前的曼哈顿距离
        goal = self.coords_to_state(self.canvas.coords(self.circle))
        curr_distance = abs(goal[0] - next_state[0]) + abs(goal[1] - next_state[1])
        prev_distance = abs(goal[0] - state[0]) + abs(goal[1] - state[1])
        # 奖励逻辑
        if self.state_map[next_state[0]][next_state[1]] == 1:
            reward, done = -100, True  # 撞到障碍物
        elif next_state == goal:
            reward, done = 100, True  # 到达终点
        else:
            reward, done = -1, False  # 普通移动基础惩罚
            # ✅ 差分奖励：如果靠近终点，奖励更高；远离则负奖励
            delta = prev_distance - curr_distance
            reward += delta * 0.5
            # ✅ 首次访问奖励：鼓励探索未知区域
            if tuple(next_state) not in self.visited:
                reward += 1
                self.visited.add(tuple(next_state))
            # ✅ 回头路惩罚：每次重复访问同一个格子都会惩罚
            self.visit_count[tuple(next_state)] += 1
            reward -= 0.2 * self.visit_count[tuple(next_state)]
            # ✅ 距离终点近时给予额外奖励（吸引力）
            if curr_distance <= 3:
                reward += (4 - curr_distance) * 2
        return next_state, reward, done

    def render(self):
       # time.sleep(0.0002) #调节速度
        self.update()
