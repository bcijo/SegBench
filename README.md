# SegBench: A Modular Segmentation Benchmarking Framework

SegBench is a flexible and modular framework for benchmarking various segmentation tasks including semantic segmentation, instance segmentation, panoptic segmentation, and image classification. The framework is designed to be easily extensible with new datasets, models, and tasks.

## Features

- **Modular Design**: Separate modules for data loading, models, utilities, and training
- **Multiple Task Support**: Semantic segmentation, instance segmentation, panoptic segmentation, and image classification
- **Flexible Architecture**: Configurable backbones and model architectures
- **Dataset Standardization**: Unified data format across different datasets
- **Easy Configuration**: YAML-based experiment configuration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/segbench.git
cd segbench
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Cityscapes
1. Register at [Cityscapes website](https://www.cityscapes-dataset.com/)
2. Download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip
3. Extract to `datasets/cityscapes/`

### PASCAL VOC
1. Download from [official website](http://host.robots.ox.ac.uk/pascal/VOC/)
2. Extract to `datasets/pascal_voc/`

### DUT-OMRON
1. Download from [official website](http://saliencydetection.net/dut-omron/)
2. Extract to `datasets/dut_omron/`

## Usage

### Training

1. Configure your experiment by modifying the YAML files in `configs/`
2. Run training:
```bash
python scripts/train.py --config configs/config_cityscapes_unet.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --config configs/config_cityscapes_unet.yaml --checkpoint path/to/checkpoint.pth
```

## Project Structure

```
.
├── datasets/               # Dataset loaders
│   ├── base_dataset.py
│   ├── cityscapes.py
│   ├── dut_omron.py
│   └── pascal_voc.py
├── models/                # Model architectures
│   ├── base_model.py
│   ├── unet.py
│   ├── pspnet.py
│   ├── hrnet.py
│   └── heads/
│       ├── semantic_head.py
│       ├── instance_head.py
│       ├── panoptic_head.py
│       └── classification_head.py
├── utils/                 # Utility functions
│   ├── transforms.py
│   ├── losses.py
│   ├── metrics.py
│   ├── setup.py
│   └── config_parser.py
├── configs/               # Configuration files
│   ├── config_cityscapes_unet.yaml
│   └── config_pascal_voc_pspnet.yaml
└── scripts/               # Training and evaluation scripts
    ├── train.py
    └── evaluate.py
```

## Adding New Components

### Adding a New Dataset
1. Create a new dataset class in `datasets/` inheriting from `BaseDataset`
2. Implement required methods: `__init__`, `__len__`, `__getitem__`
3. Add dataset-specific preprocessing in the data loader

### Adding a New Model
1. Create a new model class in `models/` inheriting from `BaseModel`
2. Implement the model architecture
3. Add configuration options in the YAML files

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
