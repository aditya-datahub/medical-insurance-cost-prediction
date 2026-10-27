# Medical Insurance Cost Prediction

Predicting individual medical insurance charges from personal attributes using **Linear Regression** in Python (scikit-learn).

## Overview

Insurers price policies based on a person's profile. This project explores a dataset of 1,338 insurance records, analyses how each factor is distributed, and trains a regression model that estimates yearly insurance charges.

## Dataset

- **File:** `insurance.csv` (1,338 rows, 7 columns, no missing values)
- **Source:** Kaggle, *Medical Cost Personal Datasets*

| Column | Description |
|---|---|
| `age` | Age of the primary beneficiary |
| `sex` | male / female |
| `bmi` | Body mass index |
| `children` | Number of dependents covered |
| `smoker` | yes / no |
| `region` | Residential area in the US (northeast, northwest, southeast, southwest) |
| `charges` | Medical costs billed by insurance (target) |

**Quick facts from the EDA**
- Sex split is balanced: 676 male, 662 female
- 274 smokers vs 1,064 non-smokers
- Regions are evenly spread (324 to 364 records each)
- Average charge is about $13,270, ranging from $1,122 to $63,770

## Approach

1. **Data checks:** shape, data types, missing values, summary statistics
2. **EDA:** distributions of age, sex, BMI, children, smoker, region and charges (Matplotlib, Seaborn)
3. **Encoding:** categorical columns converted to numbers

   | Column | Mapping |
   |---|---|
   | `sex` | male = 0, female = 1 |
   | `smoker` | yes = 0, no = 1 |
   | `region` | southeast = 0, southwest = 1, northeast = 2, northwest = 3 |

4. **Train/test split:** 80% / 20% (`random_state=42`), giving 1,070 training and 268 test rows
5. **Model:** Linear Regression
6. **Predictive system:** takes a person's details and returns an estimated charge

## Results

| Dataset | R² score |
|---|---|
| Training | 0.741 |
| Testing | 0.783 |

The test score is slightly higher than the training score, so the model is not overfitting.

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook

## How to Run

```bash
git clone https://github.com/aditya-datahub/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook main.ipynb
```

Keep `insurance.csv` in the same folder as the notebook.

## Possible Improvements

- Try Random Forest or Gradient Boosting and compare R², MAE and RMSE
- Use one-hot encoding for `region` instead of numeric labels
- Add cross-validation
- Log-transform `charges`, which is right-skewed

## Author

**Aditya Sharma**: [GitHub](https://github.com/aditya-datahub)
