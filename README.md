# Advancing Social Neuroscience: Automated Detection of Social Interaction in Movie Scenes Using Multimodal AI Models

This project aims to automate the detection of social interactions in movie scenes using various multimodal AI models, contributing to the field of social neuroscience research.

## Project Overview

This system analyzes video content to detect and classify social interactions using multiple AI models including BLIP, LLaVA, and CLIP. It compares the AI models' performance against human annotations to evaluate their effectiveness in detecting social interactions in naturalistic settings.

## Features

- Scene detection and analysis
- Multiple AI model integration (BLIP, LLaVA, CLIP)
- Support for different video formats
- Annotation comparison and metrics calculation
- Real-time video processing with annotation overlay

## Prerequisites

- Python 3.7+
- OpenCV
- PyTorch
- Transformers
- SceneDetect
- MoviePy
- SciPy

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

- `add_annotations_to_movie.py`: Adds visual annotations to video files
- `calc.py`: Calculates performance metrics (TP, FP, FN, TN, accuracy, precision, recall, F1)
- `get_annotations.py`: Handles annotation processing and model predictions
- `llmUtils.py`: Utilities for interacting with LLM models
- `movie_utils.py`: Video processing utilities
- `Scene.py`: Scene-level processing and analysis
- `test.py`: Testing and evaluation script
- `Video.py`: Video-level processing and scene management

## Usage

### Running Tests

```bash
python test.py --annotation_dir [path_to_annotations] \
               --movie_path [path_to_movie] \
               --movie_name [movie_name] \
               --pipeline \
               --blip
```

### Processing Videos with Annotations

```bash
python add_annotations_to_movie.py --annotation_dir [path_to_annotations] \
                                 --movie_path [path_to_movie] \
                                 --new_path [output_path]
```

## Models

The project implements three main AI approaches:
1. **BLIP**: Vision-language model for basic scene understanding
2. **LLaVA**: Large language model with visual capabilities
3. **CLIP**: Contrastive Language-Image Pre-training model

## Metrics

The system evaluates model performance using:
- Accuracy
- Precision
- Recall
- F1 Score

## Data Format

### Annotation Structure
Annotations are stored in `.mat` files with the following types:
- face
- written_text
- social_nonsocial
- speaking

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

[Add your license information here]

## Authors

[Your Name]

## Acknowledgments

- [Add acknowledgments for any resources, datasets, or tools used]
- Special thanks to [any specific individuals or organizations]

## Citation

If you use this work in your research, please cite:

```
[Add citation information]
```
