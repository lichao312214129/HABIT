import re

file_path = r'f:\work\habit_project\habit\core\machine_learning\machine_learning.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    # 替换简单的 print() 为 logger.info()
    (r'print\(f"Available feature selectors: {self\.available_selectors}"\)',
     r'self.logger.info("Available feature selectors: %s", self.available_selectors)'),

    (r'print\(error_msg\)',
     r'self.logger.warning(error_msg)'),

    (r'print\(f"Reading data from multiple files: {len\(self\.data_file\)} files"\)',
     r'self.logger.info("Reading data from multiple files: %d files", len(self.data_file))'),

    (r'print\(f"  Reading file: {file_path}"\)',
     r'self.logger.info("Reading file: %s", file_path)'),

    (r'print\(f"  Dataset name: {name}"\)',
     r'self.logger.info("Dataset name: %s", name)'),

    (r'print\(f"  Subject ID column: {subject_id_col}"\)',
     r'self.logger.info("Subject ID column: %s", subject_id_col)'),

    (r'print\(f"  Label column: {label_col}"\)',
     r'self.logger.info("Label column: %s", label_col)'),

    (r'print\(f"  Features: {features}"\)',
     r'self.logger.info("Features: %s", features)'),

    (r'print\(warning_msg\)',
     r'self.logger.warning(warning_msg)'),

    (r'print\(f"Adding unified label column: {self\.label_col}"\)',
     r'self.logger.info("Adding unified label column: %s", self.label_col)'),

    (r'print\(self\.data\[self\.label_col\]\.apply\(type\)\.value_counts\(\)\)',
     r'self.logger.debug("Label column types: %s", self.data[self.label_col].apply(type).value_counts())'),

    (r'print\(f"Data split using random method \(test size: {self\.test_size}\)"\)',
     r'self.logger.info("Data split using random method (test size: %s)", self.test_size)'),

    (r'print\(f"Data split using stratified method \(test size: {self\.test_size}\)"\)',
     r'self.logger.info("Data split using stratified method (test size: %s)", self.test_size)'),

    (r'print\("Converting DataFrame index to string type"\)',
     r'self.logger.info("Converting DataFrame index to string type")'),

    (r'print\(f"Data split using custom subject IDs \(train: {len\(self\.data_train\)}, test: {len\(self\.data_test\)}"\)',
     r'self.logger.info("Data split using custom subject IDs (train: %d, test: %d)", len(self.data_train), len(self.data_test))'),

    (r'print\(f"\\nExecuting {method_name} feature selection BEFORE normalization\.\.\."\)',
     r'self.logger.info("Executing %s feature selection BEFORE normalization...", method_name)'),

    (r'print\(f"Selected features: {selected_features}"\)',
     r'self.logger.info("Selected features: %s", selected_features)'),

    (r'print\(f"Skipping this method, continuing with current feature set: {len\(selected_features\)} features"\)',
     r'self.logger.info("Skipping this method, continuing with current feature set: %d features", len(selected_features))'),

    (r'print\(f"Pre-normalization feature selection completed\. Selected {len\(selected_features\)} features"\)',
     r'self.logger.info("Pre-normalization feature selection completed. Selected %d features", len(selected_features))'),

    (r'print\("Warning: No feature selection methods provided, using original feature set"\)',
     r'self.logger.warning("No feature selection methods provided, using original feature set")'),

    (r'print\(f"\\nExecuting {method_name} feature selection\.\.\."\)',
     r'self.logger.info("Executing %s feature selection...", method_name)'),

    (r'print\(f"\\nFinal number of selected features: {len\(self\.selected_features\)}"\)',
     r'self.logger.info("Final number of selected features: %d", len(self.selected_features))'),

    (r'print\(f"Selected features: {self\.selected_features}"\)',
     r'self.logger.info("Selected features: %s", self.selected_features)'),

    (r'print\(f"\\nTraining model: {model_name}"\)',
     r'self.logger.info("Training model: %s", model_name)'),

    (r'print\(error_msg\)',
     r'self.logger.error(error_msg)'),

    (r'print\(f"\\nEvaluating model: {model_name}"\)',
     r'self.logger.info("Evaluating model: %s", model_name)'),

    (r'print\(f"Feature importance: {feature_importance}"\)',
     r'self.logger.debug("Feature importance: %s", feature_importance)'),

    (r'print\(f"Saved complete prediction results to: {all_results_path}"\)',
     r'self.logger.info("Saved complete prediction results to: %s", all_results_path)'),

    (r'print\(f"Saved complete model package to: {model_package_path}"\)',
     r'self.logger.info("Saved complete model package to: %s", model_package_path)'),

    (r'print\(f"Error saving model package: {e}"\)',
     r'self.logger.error("Error saving model package: %s", e)'),

    (r'print\(f"Warning: Missing columns in new data: \{\{missing_cols\}\}"\)',
     r'self.logger.warning("Missing columns in new data: %s", missing_cols)'),

    (r'print\("No common columns found for transformation"\)',
     r'self.logger.warning("No common columns found for transformation")'),

    (r'print\(f"Successfully loaded model package from {model_file_path}"\)',
     r'self.logger.info("Successfully loaded model package from %s", model_file_path)'),
]

for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Replacement completed!")
