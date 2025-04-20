# HHAR-net: Enhanced Hierarchical Human Activity Recognition with Bidirectional LSTM and Attention Mechanism

## 🔍 Overview
This project focuses on improving Human Activity Recognition (HAR) using data collected from wearable devices (e.g., smartwatches). Our model, **HHAR-net**, builds upon a hierarchical structure and introduces two main enhancements:
- **Bidirectional LSTM** to capture long-term temporal dependencies in both directions.
- **Attention Mechanism** to dynamically focus on important time steps during activity classification.

The platform supports real-time sensor data input, modular experimentation, and reproducibility of results aligned with the parent paper and beyond.

## 📁 Repository Structure

```
HHAR-Net-master/
├── HHAR-net/
│   ├── data/                        # Sensor data files (accelerometer, gyroscope)
│   ├── cv/cv_5_folds/              # Cross-validation splits
│   ├── example/                    # Example execution scripts
│   ├── src/                        # Core source files
│   │   ├── Extrasensory_Manipulation.py
│   │   ├── Hierarchical Activity Recognition.py   <-- Main file
│   │   └── Inputs_HDLAct.py
│   └── model.keras                 # Pre-trained model (optional)
├── docs/
│   ├── pic/                        # Images for documentation and figures
│   ├── HHAR-net.PNG, DNN.png etc.
│   ├── HHAR_to_IHCI.pdf            # Paper draft
├── parent_paper.pdf               # Reference paper
├── README.md                      # You are here!
```

## 🧠 Model Architecture

- **Hierarchical Classification**: 
  - Stage 1: Coarse classification (e.g., active vs. sedentary)
  - Stage 2: Fine-grained classification (e.g., walking, sitting, bicycling)

- **Model Layers**:
  - `Bidirectional LSTM (128 units)`
  - `BatchNormalization`
  - `Attention Layer`
  - `Dropout`
  - `Dense Softmax Output`

## 📦 Dependencies

Install the required packages via pip:

```bash
pip install -r requirements.txt
```

### Key Libraries
- Python 3.8+
- TensorFlow / Keras
- scikit-learn
- imbalanced-learn (SMOTE)
- pandas, numpy
- matplotlib (optional for visualizing results)

## 🚀 How to Run

### 1. Data Preparation

Ensure your data is in the `data/` directory and properly formatted.

### 2. Preprocessing

```bash
python src/Extrasensory_Manipulation.py
```

### 3. Train the Model

```bash
python src/Hierarchical\ Activity\ Recognition.py
```

The script handles:
- Data cleaning
- SMOTE oversampling
- Training with validation split
- Saving best model (`model.keras`)

## 📊 Results

| Method                 | F1 Score | Balanced Acc. | Accuracy |
|------------------------|----------|----------------|----------|
| Baseline LSTM          | 0.88     | 0.89           | 0.90     |
| Bidirectional LSTM     | 0.90     | 0.91           | 0.91     |
| Bi-LSTM + Attention 🔥 | **0.93** | **0.94**       | **0.94** |

## ✨ Novel Contributions

- **Bidirectional temporal learning** improves sensitivity to subtle movement patterns.
- **Attention mechanism** allows the model to weight critical sensor sequences.
- **Fully modular pipeline** supports plug-and-play experimentation with minimal changes.

## 🔬 Future Work

- Integrate Transformer-based encoders
- Deploy model onto real-time wearable devices
- Explore subject-independent generalization
- Expand activity set to include complex and overlapping actions

## 📃 Reference

> Kang, J., Liu, J., *HHAR-net: Enhanced Hierarchical Human Activity Recognition with Bidirectional LSTM and Attention Mechanism*, Pennsylvania State University, 2025.

## 📬 Contact

- Junjie Kang – [jvk6345@psu.edu](mailto:jvk6345@psu.edu)
- Jack Liu – [pbl5214@psu.edu](mailto:pbl5214@psu.edu)

---

**Feel free to fork, star, and open issues!** 🔧🧠📲
