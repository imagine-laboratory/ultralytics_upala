import yaml
import argparse
from ultralytics import YOLO

def load_config(config_file):
    """Load the YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    """Command-line arguments for configuration file input."""
    parser = argparse.ArgumentParser(description="YOLOv8 Model Validation")
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the YAML configuration file (e.g., config.yaml)')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load config from the provided YAML file
    config = load_config(args.config)

    # Load the YOLO model (pretrained or provided path)
    model_path = config.get('model', 'yolov8n.pt')
    model = YOLO(model_path)

    # Set the validation parameters from the config
    val_params = {
        'data': config.get('data', 'coco8.yaml'),
        'imgsz': config.get('imgsz', 640),
        'batch': config.get('batch', 16),
        'conf': config.get('conf', 0.001),
        'iou': config.get('iou', 0.6),
        'max_det': config.get('max_det', 300),
        'half': config.get('half', True),
        'device': config.get('device', None),
        'dnn': config.get('dnn', False),
        'plots': config.get('plots', False),
        'rect': config.get('rect', False),
        'split': config.get('split', 'val'),
        'save_json': config.get('save_json', False),
        'save_hybrid': config.get('save_hybrid', False)
    }

    # Validate the model with the provided parameters
    print(f"Validating model with parameters: {val_params}")
    val_results = model.val(**val_params)

    # Print and optionally save validation results
    print(val_results)
    if val_params['save_json']:
        result_file = config.get('result_file', 'val_results.json')
        print(f"Saving validation results to {result_file}...")
        model.save(result_file)

if __name__ == "__main__":
    main()
    """
    python yolo_validate.py --config config.yaml
    """
