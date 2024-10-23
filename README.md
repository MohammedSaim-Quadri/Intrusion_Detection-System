# End-To-End ML/DL Intrusion Detection System

## Overview

This project involves developing an Intrusion Detection System (IDS) using machine learning techniques to identify and prevent network intrusions. The system leverages a neural network (NN) model trained on the CICIDS2018 dataset to classify network traffic as either normal or malicious. The project includes data preprocessing, model training, hyperparameter tuning, evaluation, and Docker containerization for deployment.

## Project Structure
├── .gitattributes ├── .gitignore ├── .dockerignore ├── folder_structure.txt ├── README.md ├── requirements.txt ├── setup.py ├── artifacts │ ├── IDS_data.csv │ ├── model_trained.pkl │ ├── preprocessor.pkl │ ├── test.csv │ └── train.csv ├── dataset │ └── train_data.csv ├── logs │ ├── 10_16_2024_17_37_52.log │ ├── 10_16_2024_17_40_53.log │ ├── 10_16_2024_18_07_53.log │ ├── 10_16_2024_18_08_11.log │ ├── 10_16_2024_18_11_08.log │ ├── 10_16_2024_18_13_28.log │ ├── 10_17_2024_16_53_16.log │ ├── 10_17_2024_16_54_53.log │ ├── 10_23_2024_18_42_52.log │ ├── 10_23_2024_19_34_16.log │ └── 10_24_2024_00_00_33.log └── src ├── exception.py ├── logger.py ├── utils.py ├── components │ ├── data_ingestion.py │ ├── data_transformation.py │ ├── model_trainer.py │ └── init.py └── init.py


## Key Features

- **Data Preprocessing**: 
  - Data ingestion and transformation processes to clean and prepare the CICIDS2018 dataset.
  - Handling missing values, encoding categorical features, and scaling numerical data.
  
- **Model Training**:
  - Implementation of a neural network for intrusion detection.
  - Training and evaluation of the model with performance metrics.

- **Hyperparameter Tuning**:
  - Utilization of Optuna for optimizing hyperparameters to enhance model performance.

- **Model Evaluation**:
  - Metrics used include accuracy, precision, recall, F1 score, and ROC AUC.

- **Docker Containerization**:
  - The project includes a Dockerfile to simplify the deployment of the IDS. This allows the application to run consistently across various environments.

## Performance Metrics

The following metrics were obtained from the trained neural network model:

- **Testing Accuracy Score**: 85.92%
- **Training Accuracy Score**: 85.89%
- **Testing F1 Score**: 82.34%
- **Training F1 Score**: 82.30%
- **Testing Precision**: 82.32%
- **Training Precision**: 82.25%
- **Testing Recall**: 85.92%
- **Training Recall**: 85.89%
- **ROC AUC (Test)**: 97.90%
- **ROC AUC (Train)**: 98.02%

These results indicate a well-performing model that generalizes effectively to unseen data, achieving high accuracy and a strong balance between precision and recall.

## Requirements

To run this project,

## Installation
Clone the repository:

```bash
git clone https://github.com/yourusername/IDS.git
cd IDS
```

## Install the required packages:

``` bash
pip install -r requirements.txt
``` 

## Running the Application:

You can run the model training directly:
```bash
python src/components/data_ingestion.py
```

## Using Docker:

Build the Docker image:
```bash
docker build -t ids-project .
```

Run the Docker container:
```bash
docker run -p 6006:6006 ids-project
```

## Future Work
- Real-Time Traffic Simulation: Implementing real-time traffic simulation for further testing and validation of the IDS.
- Improving Model Performance: Exploring advanced architectures and techniques for further optimization of the model.

## Contribution
Contributions are welcome! Please feel free to fork the repository and submit a pull request with your improvements.For major changes, please open an issue first to discuss what you would like to change.