'''https://mp.weixin.qq.com/s/LMwDl_63nE1bWK_4z46mEA
批量梯度下降可视化教学工具
'''
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 设置字体为Arial，符合顶级期刊要求
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42  # embed TrueType fonts
plt.rcParams['ps.fonttype'] = 42

class GradientDescentVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("批量梯度下降(BGD)可视化教学工具")
        self.root.geometry("1000x700")
        
        # 数据生成参数
        self.n_samples = 100
        self.n_features = 2
        
        # 随机生成模拟的训练数据
        self.generate_data()
        
        # 参数设置
        self.alpha = 0.001  # 学习率
        self.n_iter = 10  # 迭代次数
        self.current_iter = 0
        self.w_init_strategy = "random"  # 权重初始化策略：random, zeros, ones
        self.initialize_weights()  # 初始化权重
        
        # 存储训练过程中的权重和损失
        self.w_history = [self.w.copy()]
        self.loss_history = [self.compute_loss(self.w)]
        
        # 创建界面
        self.create_widgets()
        
    def generate_data(self):
        # 随机生成模拟的训练数据
        self.x_raw = np.random.randn(self.n_samples, self.n_features)
        x0 = np.ones([self.x_raw.shape[0], 1])  # 构建全为1的列向量，与特征矩阵拼接
        self.x = np.hstack([x0, self.x_raw])
        self.w_true = np.array([0.5, 0.6, -0.2]).reshape(-1, 1)  # 真实的权重向量
        self.y = np.dot(self.x, self.w_true)
    
    def initialize_weights(self):
        """根据选择的策略初始化权重"""
        if self.w_init_strategy == "random":
            self.w = np.random.randn(self.x.shape[1], 1)
        elif self.w_init_strategy == "zeros":
            self.w = np.zeros((self.x.shape[1], 1))
        elif self.w_init_strategy == "ones":
            self.w = np.ones((self.x.shape[1], 1))
        else:
            # 默认使用随机初始化
            self.w = np.random.randn(self.x.shape[1], 1)
    
    def gradient_function(self, w):
        """计算梯度
        gradient = \frac{2}{m} \{\boldsymbol{X}^T \boldsymbol{X}\boldsymbol{w} - \boldsymbol{X}^T \boldsymbol{y}\}
        """
        m = self.x.shape[0]  # 样本的个数
        gradient = 2 / m * (np.dot(self.x.T, np.dot(self.x, w)) - np.dot(self.x.T, self.y))
        return gradient
    
    def compute_loss(self, w):
        """计算损失函数值 (均方误差)"""
        m = self.x.shape[0]
        y_pred = np.dot(self.x, w)
        loss = np.sum((y_pred - self.y) ** 2) / m
        return loss
    
    def create_widgets(self):
        # 创建控制面板
        control_frame = ttk.LabelFrame(self.root, text="参数控制")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # 学习率设置
        ttk.Label(control_frame, text="学习率 (α):").grid(row=0, column=0, padx=5, pady=5)
        self.alpha_var = tk.DoubleVar(value=self.alpha)
        alpha_entry = ttk.Entry(control_frame, textvariable=self.alpha_var, width=10)
        alpha_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # 迭代次数设置
        ttk.Label(control_frame, text="迭代次数:").grid(row=0, column=2, padx=5, pady=5)
        self.iter_var = tk.IntVar(value=self.n_iter)
        iter_entry = ttk.Entry(control_frame, textvariable=self.iter_var, width=10)
        iter_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # 权重初始化策略设置
        ttk.Label(control_frame, text="权重初始化:").grid(row=0, column=4, padx=5, pady=5)
        self.init_strategy_var = tk.StringVar(value=self.w_init_strategy)
        init_combo = ttk.Combobox(control_frame, textvariable=self.init_strategy_var, 
                                values=["random", "zeros", "ones"], width=10)
        init_combo.grid(row=0, column=5, padx=5, pady=5)
        
        # 按钮
        self.reset_btn = ttk.Button(control_frame, text="重置", command=self.reset)
        self.reset_btn.grid(row=0, column=6, padx=5, pady=5)
        
        self.step_btn = ttk.Button(control_frame, text="单步训练", command=self.step)
        self.step_btn.grid(row=0, column=7, padx=5, pady=5)
        
        self.train_btn = ttk.Button(control_frame, text="完整训练", command=self.train)
        self.train_btn.grid(row=0, column=8, padx=5, pady=5)
        
        # 添加快速学习率设置按钮
        quick_alpha_frame = ttk.LabelFrame(self.root, text="快速学习率设置")
        quick_alpha_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        alpha_values = [0.0001, 0.001, 0.01, 0.1, 0.5]
        for i, alpha_val in enumerate(alpha_values):
            alpha_btn = ttk.Button(quick_alpha_frame, text=f"α={alpha_val}", 
                                  command=lambda a=alpha_val: self.set_alpha(a))
            alpha_btn.grid(row=0, column=i, padx=5, pady=5)
        
        # 创建图表区域
        chart_frame = ttk.Frame(self.root)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 设置图表
        self.fig = Figure(figsize=(10, 8), dpi=100)
        
        # 创建子图
        self.ax1 = self.fig.add_subplot(221)  # 数据点和拟合线
        self.ax2 = self.fig.add_subplot(222)  # 损失函数曲线
        self.ax3 = self.fig.add_subplot(223)  # w0 权重变化
        self.ax4 = self.fig.add_subplot(224)  # w1, w2 权重变化
        
        # 嵌入图表到Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始绘图
        self.update_plots()
        
        # 状态栏
        self.status_var = tk.StringVar(value="准备就绪")
        status_label = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
    
    def set_alpha(self, alpha_value):
        """快速设置学习率"""
        self.alpha_var.set(alpha_value)
        self.status_var.set(f"学习率已设为 {alpha_value}")
        
    def update_plots(self):
        # 清除所有子图
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # 绘制数据点和拟合线 (仅展示二维空间的投影)
        self.ax1.scatter(self.x_raw[:, 0], self.y, color='blue', alpha=0.5, label='真实数据')
        
        # 计算当前拟合线
        x_line = np.linspace(min(self.x_raw[:, 0]), max(self.x_raw[:, 0]), 100)
        w_current = self.w_history[-1]
        y_line = w_current[0] + w_current[1] * x_line.reshape(-1, 1)
        self.ax1.plot(x_line, y_line, 'r-', label='拟合线')
        
        # 绘制真实模型线
        y_true_line = self.w_true[0] + self.w_true[1] * x_line.reshape(-1, 1)
        self.ax1.plot(x_line, y_true_line, 'g--', label='真实模型')
        self.ax1.set_title('数据与拟合线')
        self.ax1.set_xlabel('特征 x1')
        self.ax1.set_ylabel('目标 y')
        self.ax1.legend()
        
        # 绘制损失函数曲线
        iterations = list(range(len(self.loss_history)))
        self.ax2.plot(iterations, self.loss_history, 'b-')
        self.ax2.set_title('损失函数')
        self.ax2.set_xlabel('迭代次数')
        self.ax2.set_ylabel('损失 (MSE)')
        
        # 绘制权重变化 - w0
        w0_history = [w[0][0] for w in self.w_history]
        self.ax3.plot(iterations, w0_history, 'g-')
        self.ax3.axhline(y=self.w_true[0], color='r', linestyle='--', label='真实值')
        self.ax3.set_title('权重 w0 变化')
        self.ax3.set_xlabel('迭代次数')
        self.ax3.set_ylabel('w0 值')
        self.ax3.legend()
        
        # 绘制权重变化 - w1, w2
        w1_history = [w[1][0] for w in self.w_history]
        w2_history = [w[2][0] for w in self.w_history]
        self.ax4.plot(iterations, w1_history, 'b-', label='w1')
        self.ax4.plot(iterations, w2_history, 'g-', label='w2')
        self.ax4.axhline(y=self.w_true[1], color='r', linestyle='--', label='w1真实值')
        self.ax4.axhline(y=self.w_true[2], color='m', linestyle='--', label='w2真实值')
        self.ax4.set_title('权重 w1, w2 变化')
        self.ax4.set_xlabel('迭代次数')
        self.ax4.set_ylabel('权重值')
        self.ax4.legend()
        
        # 调整布局
        self.fig.tight_layout()
        self.canvas.draw()
        
    def step(self):
        """执行单步梯度下降"""
        self.alpha = self.alpha_var.get()
        self.n_iter = self.iter_var.get()
        
        if self.current_iter >= self.n_iter:
            self.status_var.set(f"已完成全部 {self.n_iter} 次迭代")
            return
            
        # 计算梯度
        gradient = self.gradient_function(self.w)
        
        # 更新权重
        self.w = self.w - self.alpha * gradient
        
        # 存储权重和损失
        self.w_history.append(self.w.copy())
        self.loss_history.append(self.compute_loss(self.w))
        
        # 更新当前迭代次数
        self.current_iter += 1
        
        # 更新状态
        self.status_var.set(f"已完成 {self.current_iter}/{self.n_iter} 次迭代，当前损失: {self.loss_history[-1]:.6f}")
        
        # 更新图表
        self.update_plots()
    
    def train(self):
        """执行完整梯度下降训练"""
        self.alpha = self.alpha_var.get()
        self.n_iter = self.iter_var.get()
        
        for _ in range(self.current_iter, self.n_iter):
            # 计算梯度
            gradient = self.gradient_function(self.w)
            
            # 更新权重
            self.w = self.w - self.alpha * gradient
            
            # 存储权重和损失
            self.w_history.append(self.w.copy())
            self.loss_history.append(self.compute_loss(self.w))
            
            # 更新当前迭代次数
            self.current_iter += 1
        
        # 更新状态
        self.status_var.set(f"训练完成！共 {self.n_iter} 次迭代，最终损失: {self.loss_history[-1]:.6f}")
        
        # 更新图表
        self.update_plots()
        
        # 打印结果
        print(f"初始化策略: {self.w_init_strategy}")
        print(f"学习率: {self.alpha}")
        print(f"真实的权重向量: {self.w_true.T}")
        print(f"拟合的权重向量: {self.w.T}")
        print(f"最终损失: {self.loss_history[-1]:.6f}")
    
    def reset(self):
        """重置模型"""
        # 重新生成数据
        self.generate_data()
        
        # 获取最新的初始化策略
        self.w_init_strategy = self.init_strategy_var.get()
        
        # 重置参数
        self.current_iter = 0
        self.initialize_weights()  # 使用选定的策略初始化权重
        
        # 重置历史记录
        self.w_history = [self.w.copy()]
        self.loss_history = [self.compute_loss(self.w)]
        
        # 更新状态
        self.status_var.set(f"已重置模型，权重初始化策略: {self.w_init_strategy}")
        
        # 更新图表
        self.update_plots()


def main():
    root = tk.Tk()
    app = GradientDescentVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()