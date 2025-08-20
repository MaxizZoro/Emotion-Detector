# Emotion Detector

This project is an implementation of emotion detection using deep learning models trained on the FER-2013 dataset.  
The goal is to classify facial expressions into categories such as **happy, sad, angry, surprised, fearful, disgusted, and neutral**.

## Dataset

The dataset used is **FER-2013 (Facial Expression Recognition 2013)**, which was originally released in the [Kaggle Competition](https://www.kaggle.com/datasets/msambare/fer2013).  
It contains **35,887 grayscale, 48x48-pixel images of faces**, each labeled with one of seven emotions.

## Project Structure

```
.
├── src/                 # Source code for training and evaluation
├── models/              # Saved model files
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Emotion-Detector.git
   cd Emotion-Detector
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv emotion-env
   source emotion-env/bin/activate   # Linux/Mac
   emotion-env\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train a model (Tiny CNN or MobileNetV2), run:
```bash
python src/train.py
```

Models will be saved in the `models/` directory.

## Usage

To evaluate a trained model:
```bash
python src/evaluate.py --model models/your_model.h5
```

## Results

| Model                     | Validation Accuracy | Test Accuracy |
|----------------------------|---------------------|---------------|
| Tiny CNN (no augmentation) | ~45%                | ~44.5%        |
| Tiny CNN (with augmentation)| ~47.6%              | ~47.7%        |
| MobileNetV2 (fine-tuned)   | ~49.3%              | ~49.3%        |

## License

This project is for educational purposes.  
Dataset is provided by [Kaggle - FER-2013](https://www.kaggle.com/datasets/msambare/fer2013).
