"""
HABIT Logging System Usage Examples
====================================

This file demonstrates the correct way to use the centralized logging system.
"""

import logging
from pathlib import Path
from habit.utils.log_utils import setup_logger, get_module_logger


# ==============================================================================
# Example 1: Main Script or Application Entry Point
# ==============================================================================

def example_main_script():
    """
    Example of using logging in a main script or CLI command.
    This should be called at the entry point of your application.
    """
    # Setup logger with output directory
    output_dir = Path('./output')
    logger = setup_logger(
        name='my_app',                    # Application name
        output_dir=output_dir,            # Where to save log file
        log_filename='processing.log',    # Log file name
        level=logging.INFO                # Log level
    )
    
    # Now you can use the logger
    logger.info('Application started')
    logger.debug('This is debug info')
    logger.warning('This is a warning')
    logger.error('This is an error')
    
    # All module loggers will automatically write to the same file
    return logger


# ==============================================================================
# Example 2: Module or Utility Function
# ==============================================================================

# At module level (top of file)
logger = get_module_logger(__name__)

def example_module_function():
    """
    Example of using logging in a module.
    The logger will inherit configuration from the main script.
    """
    logger.info('Module function called')
    logger.debug('Processing data...')
    return True


# ==============================================================================
# Example 3: Class with Logging
# ==============================================================================

class ExampleProcessor:
    """
    Example of using logging in a class.
    """
    
    def __init__(self, config):
        # Get module logger in __init__
        self.logger = get_module_logger(__name__)
        self.config = config
        self.logger.info('Processor initialized')
    
    def process(self, data):
        """Process data with logging."""
        self.logger.info(f'Processing {len(data)} items')
        try:
            # Process data
            result = self._do_processing(data)
            self.logger.info('Processing completed successfully')
            return result
        except Exception as e:
            self.logger.error(f'Processing failed: {e}')
            raise
    
    def _do_processing(self, data):
        """Internal processing method."""
        self.logger.debug('Internal processing started')
        # ... do work ...
        return data


# ==============================================================================
# Example 4: Batch Processor (like BatchProcessor in habit)
# ==============================================================================

class ExampleBatchProcessor:
    """
    Example of a batch processor that sets up its own logging.
    Similar to habit.core.preprocessing.image_processor_pipeline.BatchProcessor
    """
    
    def __init__(self, config_path, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging at initialization
        self.logger = setup_logger(
            name='batch_processor',
            output_dir=self.output_dir,
            log_filename='processing.log',
            level=logging.INFO
        )
        
        self.logger.info('BatchProcessor initialized')
    
    def process_batch(self):
        """Process batch of items."""
        self.logger.info('Starting batch processing')
        # ... processing code ...
        self.logger.info('Batch processing completed')


# ==============================================================================
# Example 5: Console-Only Logging (No File)
# ==============================================================================

def example_console_only():
    """
    Example of console-only logging without saving to file.
    """
    logger = setup_logger(
        name='console_app',
        output_dir=None,  # No file logging
        level=logging.INFO
    )
    
    logger.info('This will only appear in console')
    return logger


# ==============================================================================
# Example 6: Different Log Levels
# ==============================================================================

def example_log_levels():
    """
    Example of using different log levels.
    """
    logger = setup_logger(
        name='level_test',
        output_dir=Path('./logs'),
        level=logging.DEBUG  # Will show DEBUG and above
    )
    
    logger.debug('Debug level message - detailed diagnostic info')
    logger.info('Info level message - general information')
    logger.warning('Warning level message - something to watch')
    logger.error('Error level message - something went wrong')
    logger.critical('Critical level message - serious problem')


# ==============================================================================
# Example 7: Hierarchical Loggers
# ==============================================================================

def example_hierarchical_loggers():
    """
    Example showing how hierarchical loggers work.
    """
    # Setup root logger
    main_logger = setup_logger('myapp', output_dir=Path('./output'))
    
    # Get module loggers - they will all write to the same file
    preprocessing_logger = get_module_logger('habit.preprocessing')
    habitat_logger = get_module_logger('habit.habitat')
    utils_logger = get_module_logger('habit.utils')
    
    # All these messages go to the same log file
    main_logger.info('Main application message')
    preprocessing_logger.info('Preprocessing message')
    habitat_logger.info('Habitat analysis message')
    utils_logger.info('Utils message')
    
    # In the log file, you'll see:
    # 2025-10-29 10:30:45 - habit - INFO - Main application message
    # 2025-10-29 10:30:45 - habit.preprocessing - INFO - Preprocessing message
    # 2025-10-29 10:30:45 - habit.habitat - INFO - Habitat analysis message
    # 2025-10-29 10:30:45 - habit.utils - INFO - Utils message


# ==============================================================================
# Example 8: Legacy Code Migration
# ==============================================================================

def example_legacy_migration():
    """
    Example of migrating from old logging code.
    """
    # OLD WAY (DON'T USE)
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     handlers=[
    #         logging.FileHandler('output.log'),
    #         logging.StreamHandler()
    #     ]
    # )
    # logger = logging.getLogger(__name__)
    
    # NEW WAY (USE THIS)
    logger = get_module_logger(__name__)
    
    # Everything else stays the same
    logger.info('Migrated to new logging system')


# ==============================================================================
# Complete Example: Real Application
# ==============================================================================

def real_world_example(config_path: str):
    """
    Complete example of a real application using the logging system.
    """
    import yaml
    
    # 1. Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config['out_dir'])
    
    # 2. Setup logging
    logger = setup_logger(
        name='real_app',
        output_dir=output_dir,
        log_filename='processing.log',
        level=logging.INFO
    )
    
    # 3. Log application start
    logger.info('='*60)
    logger.info('Application Started')
    logger.info('='*60)
    logger.info(f'Configuration: {config_path}')
    logger.info(f'Output directory: {output_dir}')
    
    try:
        # 4. Initialize processor (will use same logging system)
        processor = ExampleBatchProcessor(config_path, output_dir)
        
        # 5. Process data
        logger.info('Starting data processing')
        processor.process_batch()
        
        # 6. Log completion
        logger.info('='*60)
        logger.info('Application Completed Successfully')
        logger.info('='*60)
        
    except Exception as e:
        logger.error('='*60)
        logger.error('Application Failed')
        logger.error('='*60)
        logger.error(f'Error: {e}')
        logger.exception('Full traceback:')
        raise


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    # Run examples
    print("See the function definitions above for usage examples.")
    print("\nQuick Start:")
    print("1. Main script: use setup_logger()")
    print("2. Modules: use get_module_logger(__name__)")
    print("3. That's it! The system handles the rest.")

