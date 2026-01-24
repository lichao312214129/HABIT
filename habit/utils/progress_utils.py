"""
Progress bar utilities, providing custom progress bar display
"""
import os
import sys
import threading
import time

class CustomTqdm:
    """
    Custom progress bar class, used as an alternative to tqdm
    
    This class is designed to be safe for multi-processing environments.
    """
    _print_lock = threading.Lock()
    
    def __init__(self, total: int = None, desc: str = "Progress"):
        """
        Initialize progress bar
        
        Args:
            total (int, optional): Total number of iterations
            desc (str, optional): Progress bar description
        """
        self.total = total
        self.desc = desc
        self.n = 0
        self.last_print_time = 0
        self.min_print_interval = 0.1  # 最小更新间隔（秒）
        
    def update(self, n: int = 1) -> None:
        """
        Update progress bar
        
        Args:
            n (int, optional): Number of steps to update
        """
        self.n += n
        if self.total is not None:
            # 限制打印频率以避免在多进程环境中争用标准输出
            current_time = time.time()
            if current_time - self.last_print_time > self.min_print_interval or self.n >= self.total:
                self._print_progress()
                self.last_print_time = current_time
    
    def set_description(self, desc: str) -> None:
        """
        Set progress bar description
        
        Args:
            desc (str): Progress bar description
        """
        self.desc = desc
        
    def _print_progress(self) -> None:
        """Print progress bar with thread safety"""
        with CustomTqdm._print_lock:
            progress = int(self.n / self.total * 50)  # 50是进度条长度
            bar = "█" * progress + "-" * (50 - progress)
            percent = self.n / self.total * 100
            # 使用sys.stdout直接写入并刷新缓冲区
            sys.stdout.write(f"\r{self.desc}: [{bar}] {percent:.2f}% ({self.n}/{self.total})")
            sys.stdout.flush()
            if self.n >= self.total:
                sys.stdout.write("\n")
                sys.stdout.flush()