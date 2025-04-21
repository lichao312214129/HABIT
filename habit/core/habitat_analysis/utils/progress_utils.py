"""
Progress bar utilities, providing custom progress bar display
"""

class CustomTqdm:
    """
    Custom progress bar class, used as an alternative to tqdm
    """
    
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
        
    def update(self, n: int = 1) -> None:
        """
        Update progress bar
        
        Args:
            n (int, optional): Number of steps to update
        """
        self.n += n
        if self.total is not None:
            self._print_progress()
    
    def set_description(self, desc: str) -> None:
        """
        Set progress bar description
        
        Args:
            desc (str): Progress bar description
        """
        self.desc = desc
        
    def _print_progress(self) -> None:
        """Print progress bar"""
        progress = int(self.n / self.total * 50)  # 50 is the length of the progress bar
        bar = "█" * progress + "-" * (50 - progress)
        percent = self.n / self.total * 100
        print(f"\r{self.desc}: [{bar}] {percent:.2f}% ({self.n}/{self.total})", end="")
        if self.n >= self.total:
            print()

def tqdm_with_message(iterable, desc: str = "Progress", total: int = None, unit: str = "it"):
    """
    Progress bar wrapper with message, returns a CustomTqdm instance
    
    Args:
        iterable: Iterable object
        desc (str): Progress bar description
        total (int, optional): Total number of iterations, if None will be obtained from iterable
        unit (str): Unit label
        
    Returns:
        Iterator with progress bar
    """
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            total = None
            
    progress_bar = CustomTqdm(total=total, desc=desc)
    
    for item in iterable:
        yield item
        progress_bar.update(1) 