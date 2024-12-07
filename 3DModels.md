# 3D Models for Teeth Segmentation and Crown Generation  

## Overview  
This repository is part of the organization's effort to develop state-of-the-art neural networks for handling 3D data, focusing on applications in dental imaging and crown generation.  
The current focus is on **segmentation** of teeth from 3D data, with future phases planned for **crown generation** and broader dental applications.

## Features  
- **Teeth Segmentation**: Implementation of 3D segmentation models optimized for dental applications.  
- **Modular Design**: Well-organized modules for model training, evaluation, and inference.  
- **Preprocessing Tools**: Utilities for dataset cleaning, augmentation, and sampling.  
- **Visualization**: Tools to visualize outputs and segmentation results for better interpretability.  

## Directory Structure  
- **`main.py`**: Script for training and testing models.  
- **`infer_segmentation.py`**: Script for inference on new 3D input data.  
- **`requirements.txt`**: Python dependencies required to run the project.  
- **`Dataset/`**: Utilities for loading and preprocessing datasets.  
- **`models/`**: Model architectures for segmentation and generation tasks.  
- **`train/`**: Training pipelines and scripts.  
- **`losses/`**: Custom loss functions.  
- **`metrics/`**: Metrics to evaluate segmentation performance.  
- **`utils/`**: Helper functions for various tasks like logging and debugging.  
- **`config/`**: Configuration files for setting up experiments.  
- **`vis/`**: Visualization scripts and tools.  
- **`sampling/`**: Dataset augmentation and sampling methods.  
- **`images/`**: Example outputs or visualizations.

## Installation  

### Clone the Repository  
```bash  
git clone https://github.com/waleedalzamil80/3DModels.git  
cd 3DModels  
```  

### Set Up the Environment  
You can use the provided `environment.sh` script to create a virtual environment and install dependencies:  

```bash  
bash environment.sh  
```  

Alternatively, manually set up the environment:  
```bash  
python3 -m venv env  
source env/bin/activate  # On Windows: .\env\Scripts\activate  
pip install -r requirements.txt  
```

### Additional Dependencies (if needed)  
Install essential libraries:  
```bash  
pip install torch torchvision numpy scikit-learn trimesh  
```

## Usage  

### Training  
Use the `main.py` script to train a model:  
```bash  
python3 main.py \  
    --path "/path_to_dataset_folder" \  
    --test_ids "/path_to_test_ids.txt" \  
    --n_centroids 1024 \  
    --knn 16 \  
    --clean \  
    --nsamples 1 \  
    --batch_size 2 \  
    --num_workers 4 \  
    --sampling "fpsample" \  
    --p 1 \  
    --num_epochs 5 \  
    --model "PCT" \  
    --loss "crossentropy" \  
    --rigid_augmentation_train \  
    --rotat 1 \  
    --k 33  
```

### Inference  
Run inference on new data using the `infer_segmentation.py` script:  
```bash  
python3 infer_segmentation.py \  
    --model "PCT" \  
    --pretrained "/path_to_checkpoint/model_checkpoint.pth" \  
    --path "/path_to_input_file.bmesh" \  
    --clean \  
    --p 0 \  
    --sample \  
    --sampling "fpsample" \  
    --n_centroids 1024 \  
    --nsamples 16 \  
    --visualize \  
    --test \  
    --test_ids "/path_to_test_file.json" \  
    --k 33  
```

## Configuration  
- **Key Parameters**:  
  - `--path`: Path to the dataset folder or input file.  
  - `--model`: Model architecture (`DynamicGraphCNN`, `PCT`, etc.).  
  - `--loss`: Loss function (`crossentropy`, `dice`, etc.).  
  - `--sampling`: Sampling technique (`fpsample`, etc.).  
  - `--n_centroids`: Number of centroids for sampling.  
  - `--knn`: Number of nearest neighbors.  

- **Customization**: Modify `config/` files for experiment-specific settings.

## Contributing  
Contributions to the repository are welcome!  
- Fork the repository and submit pull requests for enhancements or bug fixes.  
- Open issues for questions or feature requests.  

### Current modification or bugs:
- Voxilization sampling technique `vox_fps` it gives an error so I can't use.
- In inference we don't have option to make any transformations on the data.

## Future Work  
- Develop models for **crown generation**.  
- Add support for additional 3D dental datasets.  
- Improve segmentation accuracy through fine-tuning and advanced architectures.  
- Extend visualization capabilities for better insights.
