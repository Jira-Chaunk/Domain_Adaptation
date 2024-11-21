import torch
from dann import FeatureExtractor, TaskClassifier
# Evaluate on the target domain
feature_extractor.eval()
task_classifier.eval()

with torch.no_grad():
    target_test_features = torch.tensor(target_test_features, dtype=torch.float32)
    target_test_labels = torch.tensor(target_test_labels, dtype=torch.long)

    # Predict
    target_features = feature_extractor(target_test_features)
    task_preds = task_classifier(target_features)
    predictions = torch.argmax(task_preds, dim=1)

    # Calculate accuracy
    accuracy = (predictions == target_test_labels).sum().item() / len(target_test_labels)
    print(f"Target Domain Accuracy: {accuracy * 100:.2f}%")
