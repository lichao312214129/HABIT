"""
Create mock prediction result files for model comparison testing
Utility script to generate prediction result CSV files for testing compare command
"""
import csv
from pathlib import Path

def read_subject_labels(csv_path):
    """Read subject IDs and labels from dataset CSV"""
    labels = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj_id = row['subjID']
            label = int(row['label'])
            labels[subj_id] = label
    return labels

def read_ids_from_file(file_path):
    """Read subject IDs from a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def create_prediction_file(output_path, train_ids, test_ids, labels, model_name):
    """Create a mock prediction results CSV file"""
    import random
    random.seed(42)  # For reproducibility
    
    rows = []
    
    # Create train predictions
    for subj_id in train_ids:
        label = labels.get(subj_id, 0)
        # Generate probability biased towards actual label
        if label == 1:
            prob = random.uniform(0.6, 0.95)
        else:
            prob = random.uniform(0.05, 0.4)
        pred = 1 if prob > 0.5 else 0
        
        rows.append({
            'subject_id': subj_id,
            'label': label,
            f'{model_name}_prob': round(prob, 6),
            f'{model_name}_pred': pred,
            'dataset': 'train'
        })
    
    # Create test predictions
    for subj_id in test_ids:
        label = labels.get(subj_id, 0)
        # Generate probability biased towards actual label
        if label == 1:
            prob = random.uniform(0.6, 0.95)
        else:
            prob = random.uniform(0.05, 0.4)
        pred = 1 if prob > 0.5 else 0
        
        rows.append({
            'subject_id': subj_id,
            'label': label,
            f'{model_name}_prob': round(prob, 6),
            f'{model_name}_pred': pred,
            'dataset': 'test'
        })
    
    # Write to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['subject_id', 'label', f'{model_name}_prob', f'{model_name}_pred', 'dataset']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Created {output_path} with {len(rows)} rows")

if __name__ == '__main__':
    # Update path: tests/ is now the parent, so go up one more level
    demo_data_dir = Path(__file__).parent.parent / 'demo_data' / 'ml_data'
    
    # Read data
    dataset_csv = demo_data_dir / 'breast_cancer_dataset.csv'
    train_ids_file = demo_data_dir / 'train_ids.txt'
    test_ids_file = demo_data_dir / 'test_ids.txt'
    
    labels = read_subject_labels(dataset_csv)
    train_ids = read_ids_from_file(train_ids_file)
    test_ids = read_ids_from_file(test_ids_file)
    
    # Create radiomics predictions
    radiomics_file = demo_data_dir / 'radiomics' / 'all_prediction_results.csv'
    create_prediction_file(radiomics_file, train_ids, test_ids, labels, 'LogisticRegression')
    
    # Create clinical predictions (with slightly different probabilities)
    import random
    random.seed(43)  # Different seed for different model
    clinical_file = demo_data_dir / 'clinical' / 'all_prediction_results.csv'
    create_prediction_file(clinical_file, train_ids, test_ids, labels, 'LogisticRegression')
    
    print("\n✅ Mock prediction files created successfully!")
