from kanren import run, var, lall, membero
import matplotlib.pyplot as plt

# 约束函数：确保皇后不在同一对角线
def queens_constraints(qs):
    """
    检查当前摆放的皇后是否符合八皇后规则：
    - 每个皇后都处于不同的行（递归构造保证）
    - 不能在同一条对角线上
    """
    for i in range(len(qs)):  # 遍历已放置的皇后
        for j in range(i + 1, len(qs)):  # 只比较后续皇后，避免重复
            if abs(qs[i] - qs[j]) == abs(i - j):  # 判断是否在同一对角线上
                return False
    return True

# 递归回溯函数来生成所有可能的皇后排列
def generate_permutations(N, current_permutation, possible_solutions):
    """
    递归构造所有合法的皇后排列，避免 itertools.permutations
    :param N: 棋盘大小（通常为 8）
    :param current_permutation: 当前放置的皇后排列（存储每列皇后的行索引）
    :param possible_solutions: 存储所有合法解的列表
    """
    # 终止条件：当已经放置 N 个皇后时，找到一个解
    if len(current_permutation) == N:
        possible_solutions.append(tuple(current_permutation))  # 记录找到的解
        return

    # 遍历所有可能的行位置（0 到 N-1）
    for row in range(N):
        if row not in current_permutation:  # 确保不重复使用同一行
            current_permutation.append(row)  # 选择当前行
            if queens_constraints(current_permutation):  # 剪枝：检查是否符合对角线规则
                generate_permutations(N, current_permutation, possible_solutions)  # 递归搜索下一个列
            current_permutation.pop()  # 回溯：撤销当前选择，尝试下一个可能的行

# 生成所有符合条件的皇后排列
N = 8  # 棋盘大小
possible_solutions = []  # 存储所有可能的解
generate_permutations(N, [], possible_solutions)  # 递归搜索

# 定义逻辑变量 q 并创建查询规则
q = var()
rules = lall(membero(q, possible_solutions))  # lall 作用是逻辑合取，membero 用于查找解

# 获取全部解,通过修改n的值
solutions = run(0, q, rules)

# 可视化函数
def plot_queens(solution):
    """
    绘制八皇后棋盘并显示皇后的位置
    :param solution: 由 8 个数（皇后所在的行索引）组成的元组
    """
    plt.figure(figsize=(8, 8))

    # 创建棋盘背景（黑白相间）
    board = [[(i + j) % 2 for j in range(8)] for i in range(8)]
    plt.imshow(board, cmap='binary', origin='lower', extent=(-0.5, 7.5, -0.5, 7.5))

    # 绘制网格线
    for x in range(8):
        for y in range(8):
            plt.gca().add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=2))

    # 在棋盘上绘制皇后
    for col, row in enumerate(solution):  # 遍历列索引及对应的行索引
        plt.text(col, row, '♛', fontsize=40, ha='center', va='center',
                 color='red' if (col + row) % 2 == 0 else 'blue')  # 确保皇后颜色适应背景

    plt.xticks([])  # 隐藏 x 轴刻度
    plt.yticks([])  # 隐藏 y 轴刻度
    plt.title('8 Queens Solution')  # 添加标题
    plt.show()

# 打印并可视化每个解
count = 0
for sol in solutions:
    count += 1
    print(f'Solution {count}: {sol}')
    plot_queens(sol)

print(f'Solutions found: {count}')
