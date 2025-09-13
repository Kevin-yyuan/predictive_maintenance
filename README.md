# Predictive Maintenance for Trucks

This project aims to predict the probability of a mechanical failure in a fleet of trucks using machine learning. By analyzing historical data, we can identify patterns that precede failures and proactively schedule maintenance to reduce costs and increase fleet reliability.

## Project Structure

-   `data_generation.py`: A Python script to generate the synthetic dataset for this project.
-   `predictive_maintenance_fixed.ipynb`: A Jupyter notebook that contains the complete analysis, from data loading and EDA to model training and prediction.
-   `sample_data/`: A directory containing the generated CSV files.
-   `predictive_maintenance.py`: A python script with all the code from the notebook.

## Getting Started

### Prerequisites

-   Python 3.x
-   Jupyter Notebook
-   The following Python libraries:
    -   pandas
    -   numpy
    -   matplotlib
    -   seaborn
    -   scikit-learn
    -   xgboost
    -   faker

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost faker jupyter
    ```

### How to Run

1.  **Generate the data:**
    Run the `data_generation.py` script to generate the synthetic dataset.
    ```bash
    python data_generation.py
    ```
    This will create the following files in the `sample_data` directory:
    -   `vehicle_info.csv`
    -   `maintenance_logs.csv`
    -   `sensor_data.csv`
    -   `driver_behavior.csv`
    -   `predict_set.csv`

2.  **Run the Jupyter notebook:**
    Start the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```
    Then, open the `predictive_maintenance_fixed.ipynb` file and run the cells in order.

    Alternatively, you can run the python script `predictive_maintenance.py` directly from your terminal:
    ```bash
    python predictive_maintenance.py
    ```

## Project Overview

The project is divided into the following stages:

1.  **Data Generation:** A synthetic dataset is created to simulate real-world truck data.
2.  **Exploratory Data Analysis (EDA):** The data is analyzed to understand its characteristics and identify patterns.
3.  **Data Preprocessing and Feature Engineering:** The data is cleaned, merged, and transformed to prepare it for machine learning.
4.  **Model Training:** Three different machine learning models are trained and evaluated:
    -   Logistic Regression
    -   Random Forest
    -   XGBoost
5.  **Prediction:** The best-performing model (XGBoost) is used to make predictions on a new, unseen dataset.
