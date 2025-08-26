# PreventiScan: AI Heart Disease Detector

PreventiScan is an AI-powered system designed to predict the risk of
heart disease using patient health data. Built with Python and machine
learning, this project demonstrates how data science can transform
medical decision-making and potentially save lives.

------------------------------------------------------------------------

## Overview

Heart disease remains the **#1 cause of death worldwide**, often going
undetected until it's too late. Doctors may not always have the time to
analyze every patient's health records in depth --- that's where
PreventiScan comes in.

PreventiScan analyzes patient data and classifies individuals as
either:\
- 🔵 **At Risk**\
- 🟠 **Low Risk**

The goal is simple: **help doctors make smarter, faster decisions with
the power of AI.**

------------------------------------------------------------------------

## Dataset

-   **Source:** [UCI Heart Disease
    Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
    (Cleveland Clinic records).\
-   **Records:** 297 patients.\
-   **Features (13 total):**
    -   Age
    -   Sex
    -   Chest pain type
    -   Resting blood pressure
    -   Serum cholesterol
    -   Fasting blood sugar
    -   Resting ECG results
    -   Maximum heart rate achieved
    -   Exercise-induced angina
    -   Oldpeak (ST depression)
    -   Slope of the ST segment
    -   Number of major vessels
    -   Thalassemia
-   **Target column:** Presence (1) or absence (0) of heart disease.

The dataset was cleaned using **pandas**: - Removed missing values.\
- Converted categorical features to numerical values.\
- Encoded the target variable as binary (0/1).

------------------------------------------------------------------------

## AI Model

-   **Algorithm:** Decision Tree Classifier (`scikit-learn`).\
-   **Process:**
    1.  Data preprocessed with `pandas` and `numpy`.\
    2.  Decision Tree trained to split data with yes/no questions (e.g.,
        "Is cholesterol \> 240?").\
    3.  Model visualized with **Graphviz** for interpretability.

This approach mimics how a doctor might reason about patient health but
executes decisions in **seconds**.

------------------------------------------------------------------------

## Decision Tree Visualization

The trained Decision Tree provides a clear, interpretable structure:

-   Each node = a medical question (e.g., cholesterol level, age).\
-   Each path = logical flow of decisions.\
-   Leaves:
    -   🔵 **At Risk**
    -   🟠 **Low Risk**

This ensures that predictions are **transparent and explainable**.

------------------------------------------------------------------------

## Results

-   **Tested on all 297 patients.**\
-   Predictions:
    -   **131 patients** → 🔵 At Risk\
    -   **166 patients** → 🟠 Low Risk\
-   Achieved classification based on medical features such as age,
    cholesterol, and heart rate.

⚡ **Speed:** The system predicts risk in seconds and can scale to
thousands of patients.

------------------------------------------------------------------------

## Tech Stack

-   **Programming Language:** Python\
-   **Libraries:**
    -   `pandas` -- data cleaning\
    -   `numpy` -- numerical computations\
    -   `scikit-learn` -- machine learning model training\
    -   `graphviz` -- decision tree visualization

------------------------------------------------------------------------

## References

-   Dataset: [UCI Heart Disease
    Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)\
-   Cleveland Clinic & associated hospitals for medical records.\
-   Open-source Python tools (`pandas`, `scikit-learn`, `graphviz`).

------------------------------------------------------------------------

##  Author

**Faiaz Ahmed**\
PreventiScan AI Project -- 2025

------------------------------------------------------------------------

## How to Run

1.  Clone this repository:

    ``` bash
    git clone https://github.com/your-username/preventiscan.git
    cd preventiscan
    ```

2.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Run the training script:

    ``` bash
    python train.py
    ```

4.  View results in terminal output and decision tree visualization.

------------------------------------------------------------------------

## Acknowledgments

Special thanks to:\
- **Dr. Larson** and researchers for data contributions.\
- Open-source developers of Python ML libraries.\
- UCI Machine Learning Repository.
