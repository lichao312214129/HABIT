import os
import base64
import shutil

def simple_obfuscate_file(source_file, target_file):
    """最简单的混淆方法"""
    with open(source_file, 'rb') as f:
        content = f.read()
    
    # 编码内容
    encoded = base64.b64encode(content)
    
    # 创建自解码文件
    with open(target_file, 'w') as f:
        f.write(f"""
import base64
exec(base64.b64decode({encoded!r}).decode())
""")

def simple_obfuscate_directory(source_dir, target_dir):
    """混淆整个目录"""
    os.makedirs(target_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        if rel_path == '.':
            rel_path = ''
        target_root = os.path.join(target_dir, rel_path)
        os.makedirs(target_root, exist_ok=True)
        
        for file in files:
            if file.endswith('.py'):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_root, file)
                simple_obfuscate_file(source_file, target_file)
                print(f"已混淆: {source_file} -> {target_file}")
            else:
                # 复制非Python文件
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_root, file)
                shutil.copy2(source_file, target_file)
                print(f"已复制: {source_file} -> {target_file}")

# 使用示例
simple_obfuscate_directory('habit', 'habit_dist')