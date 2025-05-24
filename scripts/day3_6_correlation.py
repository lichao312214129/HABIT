import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os
import json

def generate_synthetic_data(num_samples=100, num_features=10):
    """
    生成合成数据用于示例
    """
    # Initialize random data
    rng = np.random.RandomState(42)
    data = rng.rand(num_samples, num_features)
    
    # Add high correlation between features for demonstration
    data[:, 1] = data[:, 0] * 0.9 + rng.rand(num_samples) * 0.1  # Feature 1 highly correlated with Feature 0
    data[:, 3] = data[:, 2] * 0.92 + rng.rand(num_samples) * 0.08  # Feature 3 highly correlated with Feature 2

    # Create a DataFrame
    feature_names = [f"Feature_{i}" for i in range(num_features)]
    df = pd.DataFrame(data, columns=feature_names)
    
    return df

def run_example():
    """
    运行示例，展示功能
    """
    print("运行示例...")
    # 生成合成数据
    data = generate_synthetic_data()
    
    # 设置阈值
    threshold = 0.1
    
    # 移除高相关特征
    remaining_features = remove_highly_correlated_features(data, threshold)
    
    # 可视化结果
    visualize_correlation(data, remaining_features, "", threshold)

def detect_file_type(input_path):
    """
    自动检测文件类型
    """
    # 检查文件扩展名
    if input_path.endswith('.csv'):
        return 'csv'
    elif input_path.endswith('.xlsx') or input_path.endswith('.xls'):
        return 'excel'
    elif input_path.endswith('.parquet'):
        return 'parquet'
    elif input_path.endswith('.json'):
        return 'json'
    elif input_path.endswith('.pkl') or input_path.endswith('.pickle'):
        return 'pickle'
    
    # 如果无法从扩展名判断，尝试读取文件头部内容进行判断
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # 检查是否为CSV（通过逗号分隔）
            if ',' in first_line and len(first_line.split(',')) > 1:
                return 'csv'
            # 检查是否为JSON格式
            elif first_line.startswith('{') or first_line.startswith('['):
                return 'json'
    except:
        pass
    
    return None

def load_data(input_data, file_type=None, columns=None):
    """
    加载数据函数，支持CSV、Excel、Parquet、JSON等多种格式，以及直接接受DataFrame对象。
    支持指定列名或列范围。
    
    Args:
        input_data: 输入数据路径或DataFrame对象
        file_type: 文件类型（可选）
        columns: 列选择，可以是：
            - 列名列表，如 ['col1', 'col2']
            - 列范围字符串，如 '2:' 或 '2:10'
            - None（使用所有列）
    """
    if isinstance(input_data, pd.DataFrame):
        data = input_data
    else:
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"错误：文件 {input_data} 不存在")
        
        # 如果未指定文件类型，尝试自动检测
        if file_type is None:
            file_type = detect_file_type(input_data)
            if file_type is None:
                raise ValueError(f"无法自动检测文件类型: {input_data}")
            print(f"自动检测到文件类型: {file_type}")
        
        try:
            if file_type.lower() == 'csv':
                data = pd.read_csv(input_data)
            elif file_type.lower() == 'excel':
                data = pd.read_excel(input_data)
            elif file_type.lower() == 'parquet':
                data = pd.read_parquet(input_data)
            elif file_type.lower() == 'json':
                data = pd.read_json(input_data)
            elif file_type.lower() == 'pickle':
                data = pd.read_pickle(input_data)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
            
            
            
            # 检查数据是否为空
            if data.empty:
                raise ValueError(f"加载的数据为空: {input_data}")
                
        except Exception as e:
            raise Exception(f"加载数据错误: {e}")
    
    # 处理列选择
    if columns is not None:
        if isinstance(columns, str):
            # 处理列范围字符串
            if ':' in columns:
                start, end = columns.split(':')
                start = int(start) if start else 0
                end = int(end) if end else None
                data = data.iloc[:, start:end]
            else:
                # 单个列名
                data = data[columns]
        elif isinstance(columns, list):
            # 列名列表
            data = data[columns]
        else:
            raise ValueError("columns参数必须是列名列表或列范围字符串")
    
    return data

def remove_highly_correlated_features(data, threshold=0.9):
    # 初始化特征集合
    features = data.columns.tolist()
    
    # 迭代处理特征
    i = 0
    while i < len(features):
        current_feature = features[i]
        # 计算当前特征与后续特征的相关性
        to_remove = []
        # 计算当前特征与后续特征的相关性
        for j in range(i + 1, len(features)):
            corr = data[current_feature].corr(data[features[j]])
            if abs(corr) > threshold:
                to_remove.append(features[j])
        
        # 删除这些特征
        features = [f for f in features if f not in to_remove]
        
        # 移动到下一个特征
        i += 1
    
    return features

def visualize_correlation(data, remaining_features, outdir=None, threshold=0.9):
    """
    可视化特征相关性
    """
    # 创建图形
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    
    # 绘制移除特征前的相关性热图
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax[0])
    ax[0].set_title("Before removing highly correlated features")
    
    # 绘制移除特征后的相关性热图
    corr_matrix_after = data[remaining_features].corr()
    sns.heatmap(corr_matrix_after, annot=True, cmap="coolwarm", fmt=".2f", ax=ax[1])
    ax[1].set_title("After removing highly correlated features")
    
    # 设置旋转
    for axis in ax:
        axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
        axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
    
    # 输出移除的特征
    removed_features = [f for f in data.columns if f not in remaining_features]
    print(f"Threshold: {threshold}")
    print(f"Removed features ({len(removed_features)}): {removed_features}")
    print(f"Remaining features ({len(remaining_features)}): {remaining_features}")
    
    # 调整布局
    plt.tight_layout()
    
    # 保存结果
    if outdir:
        # 确保输出目录存在
        os.makedirs(outdir, exist_ok=True)
        
        # 保存相关性热图
        plt.savefig(os.path.join(outdir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"相关性热图已保存至: {os.path.join(outdir, 'correlation_analysis.png')}")
    else:
        plt.show()
    
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description='特征相关性分析和高相关特征删除')
    parser.add_argument('--input', type=str, help='输入数据文件路径（支持CSV、Excel、Parquet、JSON等）')
    parser.add_argument('--type', type=str, help='输入文件类型（可选，支持csv、excel、parquet、json、pickle）')
    parser.add_argument('--threshold', type=float, default=0.9, help='高相关性阈值（默认：0.9）')
    parser.add_argument('--outdir', type=str, help='输出目录路径')
    parser.add_argument('--columns', type=str, help='列选择，可以是列名列表（用逗号分隔）或列范围（如2:或2:10）')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化结果')
    parser.add_argument('--example', action='store_true', help='运行示例')
    
    args = parser.parse_args()
    
    if args.example:
        run_example()
        return
    
    try:
        # 处理列选择参数
        columns = None
        if args.columns:
            if ':' in args.columns:
                columns = args.columns
            else:
                columns = [col.strip() for col in args.columns.split(',')]
        
        # 加载数据
        data = load_data(args.input, args.type, columns)
        
        # 移除高相关特征
        remaining_features = remove_highly_correlated_features(data, args.threshold)
        
        # 保存特征列表
        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            removed_features = [f for f in data.columns if f not in remaining_features]
            remaining_df = {
                'remaining_features': remaining_features,
                'removed_features': removed_features
            }
            # change to json
            with open(os.path.join(args.outdir, 'remaining_features.json'), 'w') as f:
                json.dump(remaining_df, f, indent=4)
            print(f"特征列表已保存至: {os.path.join(args.outdir, 'remaining_features.json')}")
        
        # 可视化结果
        if args.visualize:
            visualize_correlation(data, remaining_features, args.outdir, args.threshold)
    except Exception as e:
        print(f"错误: {e}")
        print("\n使用 --example 参数运行示例:")
        print("python data3_6_correlation.py --example")

if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) == 1:
        sys.argv.extend([
            # '--input', r'F:\work\workstation_b\dingHuYingXiang\_the_third_training_202504\demo_data\results\radiomics_features_all.csv',
            # '--outdir', r'F:\work\workstation_b\dingHuYingXiang\_the_third_training_202504\demo_data\results\correlation',
            # '--threshold', '0.85',
            # '--columns', '1:'
            '--example'
        ])
    main()
