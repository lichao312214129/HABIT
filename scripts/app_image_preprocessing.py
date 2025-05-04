from habit.core.preprocessing.image_processor_pipeline import BatchProcessor
processor = BatchProcessor(config_path="./config/config_kmeans.yaml")
processor.process_batch()