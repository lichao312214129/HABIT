#!/usr/bin/env python3
"""
Code line counter for HABIT project
"""

import os
import json
import pathlib
from typing import Dict, Tuple, List
from collections import defaultdict


def count_lines_in_file(file_path: str) -> Tuple[int, int, int]:
    """
    Count lines in a single file
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        Tuple[int, int, int]: (total_lines, code_lines, comment_lines)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return 0, 0, 0
    
    total_lines = len(lines)
    code_lines = 0
    comment_lines = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:  # Empty line
            continue
        elif stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            comment_lines += 1
        else:
            code_lines += 1
    
    return total_lines, code_lines, comment_lines


def should_include_file(file_path: str, include_extensions: List[str]) -> bool:
    """
    Check if a file should be included in the count
    
    Args:
        file_path (str): Path to the file
        include_extensions (List[str]): List of extensions to include
        
    Returns:
        bool: True if file should be included
    """
    file_ext = pathlib.Path(file_path).suffix.lower()
    return file_ext in include_extensions


def count_project_lines(project_root: str = ".", include_dirs: List[str] = None, scan_all: bool = False) -> Dict:
    """
    Count lines of code in the HABIT project
    
    Args:
        project_root (str): Root directory of the project
        include_dirs (List[str]): List of directories to include (default: ['habit'])
        scan_all (bool): If True, scan all directories in project_root
        
    Returns:
        Dict: Statistics dictionary with line counts
    """
    # Define file extensions to include
    code_extensions = ['.py', '.pyx', '.pyi']
    config_extensions = ['.yaml', '.yml', '.json', '.toml', '.cfg', '.ini']
    doc_extensions = ['.md', '.rst', '.txt']
    
    # Initialize counters
    stats = {
        'directories': {},
        'file_types': {'code': {'files': 0, 'total_lines': 0, 'code_lines': 0, 'comment_lines': 0},
                      'config': {'files': 0, 'total_lines': 0, 'code_lines': 0, 'comment_lines': 0},
                      'docs': {'files': 0, 'total_lines': 0, 'code_lines': 0, 'comment_lines': 0}},
        'summary': {'total_files': 0, 'total_lines': 0, 'total_code_lines': 0, 'total_comment_lines': 0},
        'files_detail': {}  # Store detailed info for each file
    }
    
    # Directories to exclude from scanning
    exclude_dirs = {'.git', '__pycache__', '.vscode', 'node_modules', '.pytest_cache', 'results', 'test_output'}
    
    if scan_all:
        # Scan all directories in project root
        print(f"Scanning all directories in: {project_root}")
        
        # First, scan root level files
        root_files = [f for f in os.listdir(project_root) 
                     if os.path.isfile(os.path.join(project_root, f)) and not f.startswith('.')]
        
        if root_files:
            stats['directories']['root'] = {
                'files': 0, 'total_lines': 0, 'code_lines': 0, 'comment_lines': 0
            }
            
            for file in root_files:
                file_path = os.path.join(project_root, file)
                file_ext = pathlib.Path(file).suffix.lower()
                
                # Determine file type
                file_type = 'other'
                if file_ext in code_extensions:
                    file_type = 'code'
                elif file_ext in config_extensions:
                    file_type = 'config'
                elif file_ext in doc_extensions:
                    file_type = 'docs'
                
                # Count lines only for certain file types
                if file_type in ['code', 'config', 'docs']:
                    total_lines, code_lines, comment_lines = count_lines_in_file(file_path)
                    
                    # Store detailed file information
                    relative_path = os.path.relpath(file_path, project_root)
                    stats['files_detail'][relative_path] = {
                        'file_type': file_type,
                        'directory': 'root',
                        'extension': file_ext,
                        'total_lines': total_lines,
                        'code_lines': code_lines,
                        'comment_lines': comment_lines,
                        'empty_lines': total_lines - code_lines - comment_lines
                    }
                    
                    # Update file type stats
                    stats['file_types'][file_type]['files'] += 1
                    stats['file_types'][file_type]['total_lines'] += total_lines
                    stats['file_types'][file_type]['code_lines'] += code_lines
                    stats['file_types'][file_type]['comment_lines'] += comment_lines
                    
                    # Update directory stats
                    stats['directories']['root']['files'] += 1
                    stats['directories']['root']['total_lines'] += total_lines
                    stats['directories']['root']['code_lines'] += code_lines
                    stats['directories']['root']['comment_lines'] += comment_lines
                    
                    # Update summary
                    stats['summary']['total_files'] += 1
                    stats['summary']['total_lines'] += total_lines
                    stats['summary']['total_code_lines'] += code_lines
                    stats['summary']['total_comment_lines'] += comment_lines
        
        # Then scan subdirectories
        for item in os.listdir(project_root):
            item_path = os.path.join(project_root, item)
            if os.path.isdir(item_path) and item not in exclude_dirs:
                print(f"Scanning directory: {item_path}")
                
                # Initialize directory stats
                stats['directories'][item] = {
                    'files': 0, 'total_lines': 0, 'code_lines': 0, 'comment_lines': 0
                }
                
                # Walk through all files in the directory
                for root, dirs, files in os.walk(item_path):
                    # Skip certain directories
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_ext = pathlib.Path(file).suffix.lower()
                        
                        # Determine file type
                        file_type = 'other'
                        if file_ext in code_extensions:
                            file_type = 'code'
                        elif file_ext in config_extensions:
                            file_type = 'config'
                        elif file_ext in doc_extensions:
                            file_type = 'docs'
                        
                        # Count lines only for certain file types
                        if file_type in ['code', 'config', 'docs']:
                            total_lines, code_lines, comment_lines = count_lines_in_file(file_path)
                            
                            # Store detailed file information
                            relative_path = os.path.relpath(file_path, project_root)
                            stats['files_detail'][relative_path] = {
                                'file_type': file_type,
                                'directory': item,
                                'extension': file_ext,
                                'total_lines': total_lines,
                                'code_lines': code_lines,
                                'comment_lines': comment_lines,
                                'empty_lines': total_lines - code_lines - comment_lines
                            }
                            
                            # Update file type stats
                            stats['file_types'][file_type]['files'] += 1
                            stats['file_types'][file_type]['total_lines'] += total_lines
                            stats['file_types'][file_type]['code_lines'] += code_lines
                            stats['file_types'][file_type]['comment_lines'] += comment_lines
                            
                            # Update directory stats
                            stats['directories'][item]['files'] += 1
                            stats['directories'][item]['total_lines'] += total_lines
                            stats['directories'][item]['code_lines'] += code_lines
                            stats['directories'][item]['comment_lines'] += comment_lines
                            
                            # Update summary
                            stats['summary']['total_files'] += 1
                            stats['summary']['total_lines'] += total_lines
                            stats['summary']['total_code_lines'] += code_lines
                            stats['summary']['total_comment_lines'] += comment_lines
    else:
        # Original behavior: scan only specified directories
        if include_dirs is None:
            include_dirs = ['habit']
        
        # Walk through each included directory
        for include_dir in include_dirs:
            dir_path = os.path.join(project_root, include_dir)
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} does not exist")
                continue
                
            print(f"Scanning directory: {dir_path}")
            
            # Initialize directory stats
            stats['directories'][include_dir] = {
                'files': 0, 'total_lines': 0, 'code_lines': 0, 'comment_lines': 0
            }
            
            # Walk through all files in the directory
            for root, dirs, files in os.walk(dir_path):
                # Skip certain directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = pathlib.Path(file).suffix.lower()
                    
                    # Determine file type
                    file_type = 'other'
                    if file_ext in code_extensions:
                        file_type = 'code'
                    elif file_ext in config_extensions:
                        file_type = 'config'
                    elif file_ext in doc_extensions:
                        file_type = 'docs'
                    
                    # Count lines only for certain file types
                    if file_type in ['code', 'config', 'docs']:
                        total_lines, code_lines, comment_lines = count_lines_in_file(file_path)
                        
                        # Store detailed file information
                        relative_path = os.path.relpath(file_path, project_root)
                        stats['files_detail'][relative_path] = {
                            'file_type': file_type,
                            'directory': include_dir,
                            'extension': file_ext,
                            'total_lines': total_lines,
                            'code_lines': code_lines,
                            'comment_lines': comment_lines,
                            'empty_lines': total_lines - code_lines - comment_lines
                        }
                        
                        # Update file type stats
                        stats['file_types'][file_type]['files'] += 1
                        stats['file_types'][file_type]['total_lines'] += total_lines
                        stats['file_types'][file_type]['code_lines'] += code_lines
                        stats['file_types'][file_type]['comment_lines'] += comment_lines
                        
                        # Update directory stats
                        stats['directories'][include_dir]['files'] += 1
                        stats['directories'][include_dir]['total_lines'] += total_lines
                        stats['directories'][include_dir]['code_lines'] += code_lines
                        stats['directories'][include_dir]['comment_lines'] += comment_lines
                        
                        # Update summary
                        stats['summary']['total_files'] += 1
                        stats['summary']['total_lines'] += total_lines
                        stats['summary']['total_code_lines'] += code_lines
                        stats['summary']['total_comment_lines'] += comment_lines
    
    return stats


def print_statistics(stats: Dict) -> None:
    """
    Print formatted statistics
    
    Args:
        stats (Dict): Statistics dictionary from count_project_lines
    """
    print("\n" + "="*60)
    print("HABIT PROJECT CODE STATISTICS")
    print("="*60)
    
    # Print summary
    summary = stats['summary']
    print(f"\nSUMMARY:")
    print(f"  Total Files: {summary['total_files']}")
    print(f"  Total Lines: {summary['total_lines']:,}")
    print(f"  Code Lines: {summary['total_code_lines']:,}")
    print(f"  Comment Lines: {summary['total_comment_lines']:,}")
    print(f"  Empty Lines: {summary['total_lines'] - summary['total_code_lines'] - summary['total_comment_lines']:,}")
    
    # Print by file type
    print(f"\nBY FILE TYPE:")
    for file_type, data in stats['file_types'].items():
        if data['files'] > 0:
            print(f"  {file_type.upper()}:")
            print(f"    Files: {data['files']}")
            print(f"    Total Lines: {data['total_lines']:,}")
            print(f"    Code Lines: {data['code_lines']:,}")
            print(f"    Comment Lines: {data['comment_lines']:,}")
    
    # Print by directory
    print(f"\nBY DIRECTORY:")
    for dir_name, data in stats['directories'].items():
        if data['files'] > 0:
            print(f"  {dir_name}:")
            print(f"    Files: {data['files']}")
            print(f"    Total Lines: {data['total_lines']:,}")
            print(f"    Code Lines: {data['code_lines']:,}")
            print(f"    Comment Lines: {data['comment_lines']:,}")
    
    print("\n" + "="*60)


def save_to_json(stats: Dict, output_file: str = "code_statistics.json") -> None:
    """
    Save statistics to JSON file
    
    Args:
        stats (Dict): Statistics dictionary from count_project_lines
        output_file (str): Output JSON file path
    """
    # Create a clean version of stats for JSON export
    json_data = {
        'summary': stats['summary'],
        'file_types': stats['file_types'],
        'directories': stats['directories'],
        'files_detail': stats['files_detail'],
        'metadata': {
            'scan_time': str(pathlib.Path().absolute()),
            'total_scanned_files': len(stats['files_detail'])
        }
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to: {output_file}")
        print(f"Total files saved: {len(stats['files_detail'])}")
    except Exception as e:
        print(f"Error saving to JSON: {e}")


def create_files_summary_json(stats: Dict, output_file: str = "files_summary.json") -> None:
    """
    Create a simplified JSON with just filename and line counts
    
    Args:
        stats (Dict): Statistics dictionary from count_project_lines
        output_file (str): Output JSON file path
    """
    # Create simplified dictionary with just filename and line counts
    files_summary = {}
    
    for file_path, details in stats['files_detail'].items():
        files_summary[file_path] = {
            'total_lines': details['total_lines'],
            'code_lines': details['code_lines'], 
            'comment_lines': details['comment_lines'],
            'empty_lines': details['empty_lines'],
            'file_type': details['file_type']
        }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(files_summary, f, indent=2, ensure_ascii=False)
        print(f"Files summary saved to: {output_file}")
    except Exception as e:
        print(f"Error saving files summary: {e}")


def main():
    """
    Main function to run the line counter
    """
    print("Starting complete project line count...")
    
    # Count lines in all directories
    stats = count_project_lines(".", scan_all=True)
    
    # Print results
    print_statistics(stats)
    
    # Save detailed statistics to JSON
    save_to_json(stats, "code_statistics.json")
    
    # Save simplified files summary to JSON
    create_files_summary_json(stats, "files_summary.json")


if __name__ == "__main__":
    main() 