from src.preprocess import preprocess_data
from src.models.train import train_dann
from src.models.evaluate import evaluate_dann
from src.utils import load_config

if __name__ == "__main__":
    # Load configuration
    config = load_config("configs/config.yaml")
    
    # Preprocess data
    source_features, source_labels = preprocess_data(config["data"]["raw_path"] + "source.csv", "target")
    target_features, target_labels = preprocess_data(config["data"]["raw_path"] + "target.csv", "target")
    
    # Train the model
    train_dann(source_features, source_labels, target_features, config)
    
    # Evaluate the model
    evaluate_dann(target_features, target_labels, config)
