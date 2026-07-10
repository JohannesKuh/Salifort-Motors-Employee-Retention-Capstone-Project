# Salifort Motors - Employee Retention Analysis
Predicting employee turnover at Salifort Motors - Capstone project for the Google Advanced Data Analytics Certificate 

## Project Overview
Analyzed survey data from 14,999 employees to predict turnover 
using tree-based ML models, as part of the Google Advanced Data 
Analytics Certificate capstone.

## Business Problem
Senior leadership wanted to understand why employees leave and 
build a model to predict attrition, enabling proactive retention efforts.

## Dataset
- Dataset: `HR_capstone_dataset.csv`
- Link: [Kaggle HR Analytics Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction)
- 14,999 rows, 10 features
- Key variables: satisfaction level, number of projects, 
  average monthly hours, tenure, department, salary band

| Variable | Description |
|---|---|
| `satisfaction_level` | Employee-reported job satisfaction level [0–1] |
| `last_evaluation` | Score of employee's last performance review [0–1] |
| `number_project` | Number of projects employee contributes to |
| `average_monthly_hours` | Average number of hours employee worked per month |
| `time_spend_company` | How long the employee has been with the company (years) |
| `work_accident` | Whether or not the employee experienced an accident while at work |
| `left` | Whether or not the employee left the company |
| `promotion_last_5years` | Whether or not the employee was promoted in the last 5 years |
| `department` | The employee's department |
| `salary` | The employee's salary level (low / medium / high) |

## Approach
- Exploratory Data Analysis (EDA)
- Feature engineering & preprocessing
- Model: [Random Forest / XGBoost] with GridSearchCV tuning
- Evaluation: F1 score, precision, recall, confusion matrix

## Key Results
- Champion model: Random Forest with F1 = 0.95 on the test set
- Top predictors: satisfaction level, number of projects, tenure, average monthly hours, last evaluation

## Tools
Python, pandas, scikit-learn, XGBoost, seaborn, matplotlib, MLflow, Claude (Anthropic)

## MLflow Experiment Tracking

All experiments are tracked using MLflow 3.14 with a SQLite backend 
persisted to Google Drive.

| Run | F1 (Left) | Recall | Precision | Accuracy |
|---|---|---|---|---|
| Random Forest | 0.95 | 0.93 | 0.98 | 0.99 |
| XGBoost | 0.95 | 0.93 | 0.97 | 0.98 |
| Decision Tree | 0.94 | 0.93 | 0.95 | 0.98 |
| Logistic Regression | 0.59 | 0.87 | 0.44 | 0.80 |

![Model Registry](images/06_model_registry.png)
![Random Forest Run](images/03_runs_random_forest.png)

## Acknowledgements
- Dataset is part of the Advanced Data Analytics Certificate on Coursera, publicly available on Kaggle (see Dataset > Link)
- AI assistance provided by Claude (Anthropic) for code guidance, interpretation refinement and documentation support
