# Synthetic Data Generation with CTGAN and Evaluation

This project focuses on generating synthetic data using **CTGAN** and other generative models, and evaluates the quality using advanced techniques like **FID (Fréchet Inception Distance)**.

## 📌 Project Objectives

- Train **CTGAN** on structured/tabular data
- Generate synthetic samples
- Evaluate synthetic data quality using pretrained models (InceptionV3)
- Measure the distributional similarity between real and synthetic data

## 🧰 Technologies & Libraries

- Python, PyTorch
- CTGAN (from `sdv`)
- torchvision, PIL, NLTK
- pandas, numpy
- Pretrained InceptionV3 for FID evaluation

## 🗂️ Structure

- `ctgan-2 final.ipynb` — Main notebook containing the full workflow.
- Custom Dataset class — Used to handle structured image-text or tabular data.
- InceptionV3 Network — Used for feature extraction during evaluation.

## ⚙️ Setup Instructions

1. **Clone this repo** and install dependencies:

    ```bash
    git clone https://github.com/dheerubaba123/ctgan-project.git
    cd ctgan-project
    pip install -r requirements.txt
    ```

2. **Run Jupyter Notebook**

    ```bash
    jupyter notebook ctgan-2\ final.ipynb
    ```

## 📊 Evaluation

- Evaluation includes visual inspection and FID score calculation.
- Uses pretrained **InceptionV3** to compute feature embeddings of real and fake samples.

## 📦 Dependencies

```text
torch
torchvision
numpy
pandas
nltk
Pillow
sdv
