import torch
from dann import FeatureExtractor, TaskClassifier, DomainClassifier
from preprocess import preprocess_data
# Initialize models
input_dim = source_features.shape[1]
feature_dim = 64
num_classes = 2  # Example: 2 classes for cyclone severity
feature_extractor = FeatureExtractor(input_dim, feature_dim)
task_classifier = TaskClassifier(feature_dim, num_classes)
domain_classifier = DomainClassifier(feature_dim)

# Loss functions and optimizers
task_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCELoss()
optimizer = optim.Adam(
    list(feature_extractor.parameters()) + list(task_classifier.parameters()) + list(domain_classifier.parameters()),
    lr=0.001
)

# Training loop
num_epochs = 50
batch_size = 32
lambda_adapt = 0.1  # Trade-off parameter for domain adaptation

def get_batch(features, labels, batch_size):
    idx = np.random.choice(len(features), batch_size, replace=False)
    return torch.tensor(features[idx], dtype=torch.float32), torch.tensor(labels[idx], dtype=torch.long)

for epoch in range(num_epochs):
    feature_extractor.train()
    task_classifier.train()
    domain_classifier.train()

    # Source domain batches
    source_batch_features, source_batch_labels = get_batch(source_features, source_labels, batch_size)
    # Target domain batches (unlabeled)
    target_batch_features, _ = get_batch(target_features, target_labels, batch_size)

    # Extract features
    source_features = feature_extractor(source_batch_features)
    target_features = feature_extractor(target_batch_features)

    # Task classification loss (source domain only)
    task_preds = task_classifier(source_features)
    task_loss = task_criterion(task_preds, source_batch_labels)

    # Domain classification loss (both source and target domains)
    domain_preds_source = domain_classifier(source_features.detach())
    domain_preds_target = domain_classifier(target_features.detach())
    domain_labels_source = torch.zeros(batch_size, 1)
    domain_labels_target = torch.ones(batch_size, 1)
    domain_loss = domain_criterion(domain_preds_source, domain_labels_source) + \
                  domain_criterion(domain_preds_target, domain_labels_target)

    # Combined loss
    total_loss = task_loss + lambda_adapt * domain_loss

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Task Loss: {task_loss.item():.4f}, Domain Loss: {domain_loss.item():.4f}")
