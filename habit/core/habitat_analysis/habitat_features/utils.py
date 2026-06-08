# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
import pandas as pd
import os

def flatten_dict(data):
    """
    将嵌套字典扁平化为指定格式的字典。
    例如：
    输入：{1: {'num_regions': 23, 'volume_ratio': 0.49297752808988765}, 3: {'num_regions': 5, 'volume_ratio': 0.5070224719101124}, 'num_habitats': 2}
    输出：{'num_regions_1': 23, 'volume_ratio_1': 0.49297752808988765, 'num_regions_3': 5, 'volume_ratio_3': 0.5070224719101124, 'num_habitats': 2}
    """
    if not isinstance(data, dict):
        raise ValueError("输入必须是一个字典（dict）。")

    flat_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):  # 如果值是字典，进一步展开
            for sub_key, sub_value in value.items():
                flat_dict[f"{sub_key}_{key}"] = sub_value
        else:  # 如果值不是字典，直接添加到结果中
            flat_dict[key] = value
    return flat_dict

def save_to_excel_sheet(df, file_name, sheet_name):
    """
    将 DataFrame 写入 Excel 文件：
    - 若文件存在：覆盖指定 Sheet，保留其他 Sheet
    - 若文件不存在：创建新文件并写入 Sheet
    """
    try:
        # 尝试追加模式（文件已存在）
        with pd.ExcelWriter(file_name, engine='openpyxl', mode='a') as writer:
            # 检查目标 Sheet 是否存在
            if sheet_name in writer.book.sheetnames:
                # 删除旧 Sheet
                writer.book.remove(writer.book[sheet_name])
            # 写入新 Sheet
            df.to_excel(writer, sheet_name=sheet_name, index=True)
            print(f"✅ 数据已覆盖写入文件 {file_name} 的 Sheet [{sheet_name}]")
            
    except FileNotFoundError:
        # 文件不存在，创建新文件
        with pd.ExcelWriter(file_name, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=True)
            print(f"🆕 文件 {file_name} 不存在，已创建并写入 Sheet [{sheet_name}]")
            
    except Exception as e:
        print(f"❌ 保存失败，错误：{str(e)}")



