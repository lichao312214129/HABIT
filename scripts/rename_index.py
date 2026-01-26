"""
给定文件夹，对文件夹下面的csv和xlsx文件的第一列进行处理
即提取第一个连续的字数字符串，将其作为index
替换第一列，并把第一列的header改为index
"""

import os
import re
import pandas as pd
import argparse
from pathlib import Path


def extract_alphanumeric_string(text):
    """
    从文本中提取第一个连续的字数字符串
    
    Args:
        text: 输入文本
        
    Returns:
        提取到的字数字符串，如果没有找到则返回原文本
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    # 使用正则表达式匹配第一个连续的字数字符串
    # Use regular expression to match the first continuous digit string
    # 匹配第一个连续的数字字符串
    match = re.search(r'\d+', text)
    if match:
        return match.group()
    return text


def process_file(file_path):
    """
    处理单个文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        处理后的DataFrame
    """
    print(f"处理文件: {file_path}")
    
    # 根据文件扩展名读取文件
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        print(f"不支持的文件格式: {file_path}")
        return None
    
    if df.empty:
        print(f"文件为空: {file_path}")
        return df
    
    # 获取第一列的名称
    first_col_name = df.columns[0]
    
    # 对第一列应用提取函数
    df[first_col_name] = df[first_col_name].apply(extract_alphanumeric_string)
    
    # 将第一列设置为索引
    df.set_index(first_col_name, inplace=True)
    
    # 重置索引，使第一列重新成为普通列，并命名为"index"
    df.reset_index(inplace=True)
    df.rename(columns={first_col_name: 'index'}, inplace=True)
    
    return df


def save_file(df, original_path, output_dir=None):
    """
    保存处理后的文件
    
    Args:
        df: 处理后的DataFrame
        original_path: 原始文件路径
        output_dir: 输出目录，如果为None则覆盖原文件
    """
    if output_dir:
        output_path = Path(output_dir) / original_path.name
    else:
        output_path = original_path
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 根据文件扩展名保存
    if original_path.suffix.lower() == '.csv':
        df.to_csv(output_path, index=False)
    elif original_path.suffix.lower() in ['.xlsx', '.xls']:
        df.to_excel(output_path, index=False)
    
    print(f"文件已保存: {output_path}")


def process_folder(folder_path, output_dir=None):
    """
    处理文件夹中的所有CSV和XLSX文件
    
    Args:
        folder_path: 文件夹路径
        output_dir: 输出目录，如果为None则覆盖原文件
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"文件夹不存在: {folder_path}")
        return
    
    # 查找所有CSV和XLSX文件
    supported_extensions = ['.csv', '.xlsx', '.xls']
    files_to_process = []
    
    for ext in supported_extensions:
        files_to_process.extend(folder_path.glob(f"*{ext}"))
    
    if not files_to_process:
        print(f"在文件夹 {folder_path} 中没有找到CSV或XLSX文件")
        return
    
    print(f"找到 {len(files_to_process)} 个文件需要处理")
    
    # 处理每个文件
    for file_path in files_to_process:
        try:
            df = process_file(file_path)
            if df is not None:
                save_file(df, file_path, output_dir)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="处理CSV和XLSX文件的第一列，提取字数字符串作为索引")
    parser.add_argument('folder', help='要处理的文件夹路径')
    parser.add_argument('--output', '-o', help='输出目录，如果不指定则覆盖原文件')
    
    args = parser.parse_args()
    
    # 如果没有命令行参数，使用默认文件夹
    if len(sys.argv) == 1:
        print("调试模式：使用当前目录")
        folder_path = "."
    else:
        folder_path = args.folder
    
    process_folder(folder_path, args.output)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            'H:\\results\\features'
        ])
    main()
    # python scripts/rename_index.py H:\results_icc_remapping --output H:\results\results_icc_remapping_index

