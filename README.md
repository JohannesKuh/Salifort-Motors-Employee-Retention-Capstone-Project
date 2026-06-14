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
- Source: Google Advanced Data Analytics Certificate program (via Coursera)
- 14,999 rows, 10 features
- Key variables: satisfaction level, number of projects, 
  average monthly hours, tenure, department, salary band

  | Column | Type | Description |
|---|---|---|
| `satisfaction_level` | float | Self-reported satisfaction level [0–1] |
| `last_evaluation` | float | Last performance review score [0–1] |
| `number_project` | int | Number of projects contributed to |
| `average_monthly_hours` | int | Average hours worked per month |
| `time_spend_company` | int | Tenure in years |
| `work_accident` | int | Experienced a work accident (0/1) |
| `left` | int | Left the company — **target variable** (0/1) |
| `promotion_last_5years` | int | Promoted in the last 5 years (0/1) |
| `department` | str | Employee's department |
| `salary` | str | Salary band (low / medium / high) |

## Approach
- Exploratory Data Analysis (EDA)
- Feature engineering & preprocessing
- Model: [Random Forest / XGBoost] with GridSearchCV tuning
- Evaluation: F1 score, precision, recall, confusion matrix

## Key Results
- Champion model: [your model] with F1 = [your score]
- Top predictors: [e.g. satisfaction level, number of projects, tenure]

## Tools
Python, pandas, scikit-learn, XGBoost, seaborn, matplotlib

## File Structure
├── salifort_capstone.ipynb   # Main analysis notebook
├── HR_capstone_dataset.csv   # Employee survey data
└── README.md
