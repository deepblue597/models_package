# Models Package

[![PyPI version](https://badge.fury.io/py/models-package.svg)](https://badge.fury.io/py/models-package)
[![Python](https://img.shields.io/pypi/pyversions/models-package.svg)](https://pypi.org/project/models-package/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python package for water and river segmentation using deep learning models with satellite imagery data.

## 🌊 Features

- **Abstract Base Model**: Extensible architecture for different segmentation models
- **Water Segmentation**: Advanced model using Sentinel-1, Sentinel-2, and terrain data
- **River Segmentation**: Specialized model for river detection and analysis
- **Data Integration**: Built-in support for MinIO, TimescaleDB, and Kafka streaming
- **Data Validation**: Pydantic schemas for robust data handling
- **Production Ready**: Optimized for real-world deployment scenarios

## 📦 Installation

```bash
pip install models-package
```

### Development Installation

```bash
pip install models-package[dev]
```

## 🚀 Quick Start

### Water Segmentation

```python
from models_package import WaterSegmentationModel

# Initialize the water segmentation model
model = WaterSegmentationModel(
    model_path="path/to/model.pth",
    model_name="WaterSegmentation",
    model_indx=0  # 0 for 7-band model, 1 for 9-band model
)

# Load the model weights
model.load_model()

# Setup connections (optional)
model.minIOConnection(
    address="localhost",
    port=9000,
    target="bucket-name",
    access_key="access_key",
    secret_key="secret_key"
)

# Make predictions
result = model.predict({
    "folder_link": "path/to/satellite/data",
    "location": "area_name"
})

print(f"Water coverage: {result['water_coverage_stats']['water_percentage']:.2f}%")
```

### River Segmentation

```python
from models_package import RiverSegmentationModel

# Initialize the river segmentation model
model = RiverSegmentationModel(
    model_path="path/to/river_model.pth",
    model_name="RiverSegmentation",
    input_size=(512, 512),
    model_architecture="unetplusplus",
    encoder_name="efficientnet-b3"
)

# Load model and make predictions
model.load_model()
result = model.predict({
    "image_link": "path/to/image",
    "filename": "river_image.jpg"
})
```

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- CUDA compatible GPU (recommended)
- See `pyproject.toml` for complete dependency list

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Author**: Jason
- **Email**: ikakandris@gmail.com
- **Repository**: [https://github.com/deepblue597/models_package](https://github.com/deepblue597/models_package)

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- See requirements.txt for full dependencies

## License

MIT License
