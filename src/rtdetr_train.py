import yaml
import argparse
from ultralytics import RTDETR

def load_config(config_file):
    """Load the YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    """Command-line arguments for configuration file input."""
    parser = argparse.ArgumentParser(description="YOLOv8 Model Training and Validation")
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the YAML configuration file (e.g., config.yaml)')
    return parser.parse_args()

def freeze_model_layers(model, num_layer=10):
    """Function to freeze model layers."""
    freeze = [f"model.{x}." for x in range(num_layer+1)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f"freezing {k}")
            v.requires_grad = False

def print_model_layers_grads(model, num_layer=10):
    """Function to print model layers grads."""
    freeze = [f"model.{x}." for x in range(num_layer+1)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f"Layer: {k} - Required grad: {v.requires_grad}")
            
def main():
    # Parse command-line arguments
    args = parse_args()

    # Load config from the provided YAML file
    config = load_config(args.config)

    # Train mode of the model 
    train_mode = config.get('training_mode', 'fine-tuning')
    if train_mode == 'fine-tuning':
        # Load the YOLO model (pretrained or fine tuning)
        model_path = config.get('model', 'rtdetr-l.pt')
        model = RTDETR(model_path)

    elif train_mode == 'scratch':
        # Load the YOLO model .yaml file for scratch training
        model_path = config.get('model_yaml', 'rtdetr-l.yaml')
        model = RTDETR(model_path)

    elif train_mode == 'freeze-backbone': 
        # Load the YOLO model .yaml file for scratch training
        model_path = config.get('model', 'rtdetr-l.pt')
        num_backbone_layers = config.get('freeze_backbone_layers', 10)
        model = RTDETR(model_path)
        freeze_model_layers(model, num_layer=num_backbone_layers)
        print_model_layers_grads(model, num_layer=num_backbone_layers)

    elif train_mode == 'freeze-all': 
        # Load the YOLO model .yaml file for scratch training
        model_path = config.get('model', 'rtdetr-l.pt')
        num_all_layers = config.get('freeze_all_layers', 22)
        model = RTDETR(model_path)
        freeze_model_layers(model, num_layer=num_all_layers)
        print_model_layers_grads(model, num_layer=num_all_layers)

    # Set the training parameters from the config
    train_params = {
        'data': config.get('data', 'coco8.yaml'),
        'epochs': config.get('epochs', 3),
        'patience': config.get('patience', 100),
        'batch': config.get('batch', 16),
        'imgsz': config.get('imgsz', 640),
        'save': config.get('save', True),
        'save_period': config.get('save_period', -1),
        'cache': config.get('cache', False),
        'device': config.get('device', None),
        'workers': config.get('workers', 8),
        'project': config.get('project', './yolo_weights'),
        'name': f"{config.get('name', 'none')}@{config.get('training_mode', 'fine-tuning')}" ,
        'exist_ok': config.get('exist_ok', True),
        'pretrained': config.get('pretrained', True),
        'optimizer': config.get('optimizer', 'auto'),
        'verbose': config.get('verbose', False),
        'seed': config.get('seed', 0),
        'deterministic': config.get('deterministic', True),
        'single_cls': config.get('single_cls', False),
        'rect': config.get('rect', False),
        'cos_lr': config.get('cos_lr', False),
        'close_mosaic': config.get('close_mosaic', 10),
        'resume': config.get('resume', False),
        'amp': config.get('amp', True),
        'fraction': config.get('fraction', 1.0),
        'profile': config.get('profile', False),
        'lr0': config.get('lr0', 0.01),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'warmup_epochs': config.get('warmup_epochs', 3.0),
        'warmup_momentum': config.get('warmup_momentum', 0.8),
        'warmup_bias_lr': config.get('warmup_bias_lr', 0.1),
        'box': config.get('box', 7.5),
        'cls': config.get('cls', 0.5),
        'dfl': config.get('dfl', 1.5),
        'pose': config.get('pose', 12.0),
        'kobj': config.get('kobj', 2.0),
        'label_smoothing': config.get('label_smoothing', 0.0),
        'nbs': config.get('nbs', 64),
        'overlap_mask': config.get('overlap_mask', True),
        'mask_ratio': config.get('mask_ratio', 4),
        'dropout': config.get('dropout', 0.0),
        'val': config.get('val', True),
        'plots': config.get('plots', False),
    }

    # Train the model with the provided parameters
    print(f"Training model with parameters: {train_params}")
    model.train(**train_params)

    # Validate the model
    print("Validating the model...")
    val_results = model.val()

    # Save the model if needed
    if train_params['save']:
        save_path = config.get('save_path', 'best_model.pt')
        print(f"Saving the model to {save_path}...")
        model.save(save_path)

if __name__ == "__main__":
    main()
    """
    python yolo_script.py --config config.yaml
    """