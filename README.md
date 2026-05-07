# Chennai House Price Prediction using Ensemble Learning

A machine learning project that predicts house prices in Chennai using multiple ensemble learning techniques such as Random Forest, Gradient Boosting, XGBoost, LightGBM, and Stacking Ensemble.

The project compares different regression models, evaluates their performance using multiple metrics, visualizes results with detailed plots, and saves the best-performing model for future predictions.

---

## Features

- Data preprocessing and feature scaling
- Label encoding for categorical features
- Multiple ensemble regression models:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - LightGBM Regressor
  - Stacking Ensemble
- Performance evaluation using:
  - R² Score
  - RMSE
  - MAE
  - MAPE
  - Cross-validation R²
- Advanced visualization dashboard
- Automatic best model selection
- Model serialization using Pickle

---

## Tech Stack

- Python
- Scikit-learn
- XGBoost
- LightGBM
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Dataset

The project uses the **Housing Prices in Chennai** dataset containing various housing-related attributes and target price values.

### Example Features
- Area
- Bedroom count
- Bathroom count
- Parking
- Location
- Build size
- Other housing attributes

---

## Models Implemented

| Model | Description |
|---|---|
| Random Forest | Ensemble of decision trees using bagging |
| Gradient Boosting | Sequential boosting-based regression |
| XGBoost | Optimized gradient boosting framework |
| LightGBM | Fast gradient boosting algorithm |
| Stacking Ensemble | Combines all base models using Ridge Regression |

---

## Evaluation Metrics

The models are evaluated using:

- **R² Score**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**
- **5-Fold Cross Validation**

---

## Visualizations

The project generates a complete performance dashboard including:

- R² comparison chart
- RMSE comparison chart
- Cross-validation scores
- Actual vs Predicted scatter plots
- Residual distribution plots
- Feature importance analysis
- Final metrics summary table

---

## Project Structure

```bash
├── ensemble.py
├── best_model.pkl
├── outputs/
│   └── ensemble_report.png
├── dataset/
│   └── Housing-Prices-in-Chennai.csv
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <your-github-repo-link>
cd <repo-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```

---

## Run the Project

```bash
python ensemble.py
```

---

## Output

After execution, the project will:

- Train all ensemble models
- Compare performance metrics
- Generate visual analytics
- Save the best-performing model as:

```bash
best_model.pkl
```

- Save visualization report:

```bash
outputs/ensemble_report.png
```

---

## Best Model Selection

The script automatically selects the model with the highest R² score and stores:

- Trained model
- Scaler
- Label encoder
- Feature list

using Pickle serialization.

---

## Future Improvements

- Deploy as a web application using Flask or Streamlit
- Hyperparameter optimization
- Real-time Chennai property price prediction API
- Feature engineering improvements
- Deep learning-based regression comparison

---

## Learning Outcomes

This project demonstrates:

- Ensemble learning techniques
- Regression model comparison
- Model stacking
- Cross-validation
- Feature importance analysis
- Data preprocessing pipeline
- Model persistence

---

## Author

**Venkateswar**

Machine Learning & AI Enthusiast
