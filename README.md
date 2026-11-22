# Syntecxhub_House_Price_Prediction
> **SyntecxHub Machine Learning Internship Program — Week 1 Task**  
> **Project 1: House Price Prediction using Linear Regression**  
> *Author:* Machine Learning Intern  
> *Organization:* [SyntecxHub](https://syntecxhub.com) (*Create | Think | Solve*)

---

## 📌 Project Overview & PDF Task Constraints

This repository contains the complete implementation of **Project 1: House Price Prediction** adhering strictly to all requirements and constraints outlined in the SyntecxHub Machine Learning Internship Program:

| # | Task Constraint (from PDF Slide 5) | Implementation in Repository |
|---|---|---|
| **1** | **Load a housing dataset, clean and explore features** | Loaded in `src/data_preprocessing.py`, missing values imputed, duplicates removed, and comprehensive EDA in `src/eda.py` |
| **2** | **Select features, split train/test and train a Linear Regression model** | 8 domain features selected, 80/20 train/test split, Ordinary Least Squares (OLS) trained in `src/train.py` |
| **3** | **Evaluate using RMSE and $R^2$, and interpret coefficients** | Evaluated on Train and Test sets ($R^2 = 0.9811$, $\text{RMSE} = \$17,190.29$); marginal coefficients explained in economic dollar impact |
| **4** | **Save model and show example predictions** | Model serialized with `joblib` into `models/house_price_model.pkl`, evaluated with 4 distinct real-world home archetypes in `src/predict.py` |
| **5** | **Repository Naming Rule (PDF Slide 4)** | Repository named: `Syntecxhub_House_Price_Prediction` |
| **6** | **LinkedIn Internship Mention** | Ready-to-post LinkedIn announcement template mentioning `@Syntecxhub` provided below |

---

## 🏗️ Project Architecture & Directory Structure

```
Syntecxhub_House_Price_Prediction/
├── data/
│   ├── housing_data.csv               # Raw dataset with realistic market noise & missing values
│   └── cleaned_housing_data.csv       # Preprocessed, imputed & engineered dataset
├── models/
│   ├── house_price_model.pkl          # Serialized scikit-learn Linear Regression model
│   └── model_metadata.json            # Model parameters, evaluation metrics & coefficient dictionary
├── notebooks/
│   └── House_Price_Prediction.ipynb   # Interactive step-by-step Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── generate_data.py               # Generates synthetic housing data adhering to market economics
│   ├── data_preprocessing.py          # Data cleaning, null imputation, outlier handling & feature engineering
│   ├── eda.py                         # Statistical distributions, correlation heatmaps & visualizations
│   ├── train.py                       # Train/test split, OLS model fitting, evaluation & artifact persistence
│   └── predict.py                     # Inference script displaying novel example predictions & breakdowns
├── static/
│   ├── css/
│   │   └── style.css                  # Custom stylesheet for the interactive Flask dashboard
│   └── plots/                         # Generated diagnostic plots
│       ├── price_distribution.png     # Target price distribution with mean & median markers
│       ├── correlation_heatmap.png    # Feature correlation heatmap
│       ├── sqft_vs_price.png          # Living area vs sale price scatter plot with regression line
│       ├── quality_vs_price.png       # Box plot of house price across quality ratings
│       ├── actual_vs_predicted.png    # Actual vs predicted scatter plot validating test fit
│       ├── residuals_distribution.png # Distribution of residuals verifying regression assumptions
│       └── feature_importance.png     # Bar chart of marginal regression coefficients
├── templates/
│   └── index.html                     # Responsive Flask web interface with live property estimator
├── app.py                             # Flask web server for interactive browser predictions
├── requirements.txt                   # Project Python dependencies
├── run_pipeline.py                    # Master one-click script running end-to-end workflow
└── README.md                          # Project documentation and submission details
```

---

## ⚙️ Installation & Quickstart

### 1. Clone or Open the Repository
```bash
git clone https://github.com/<your-username>/Syntecxhub_House_Price_Prediction.git
cd Syntecxhub_House_Price_Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Entire End-to-End Pipeline
Execute the master pipeline to clean data, train the model, evaluate metrics, generate plots, and run sample inferences in one step:
```bash
python run_pipeline.py
```

### 4. Launch the Interactive Web Dashboard
Run the Flask web application to estimate house prices in real time through a browser UI:
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 📊 Dataset & Feature Selection

The dataset contains 1,600+ property observations with the following selected features for modeling:

| Feature Name | Type | Description | Unit / Range |
|---|---|---|---|
| `Square_Feet` | Continuous | Finished interior living area | 650 – 4,800 sq ft |
| `Bedrooms` | Discrete | Total number of bedrooms | 1 – 6 bedrooms |
| `Bathrooms` | Continuous | Total full and half bathrooms | 1.0 – 4.5 baths |
| `House_Age` | Continuous | Age of structure (`2024 - Year_Built`) | 0 – 70 years |
| `Garage_Cars` | Discrete | Enclosed garage vehicle capacity | 0 – 3 cars |
| `Lot_Area` | Continuous | Total land lot parcel size | 2,000 – 22,000 sq ft |
| `Overall_Quality`| Ordinal | Overall material finish & build grade | 1 (Poor) – 10 (Luxury) |
| `Neighborhood_Score`| Continuous | Location desirability index | 1.0 – 10.0 index |
| **`Price`** (Target) | Continuous | Recorded house sale price | \$60,000 – \$1,200,000+ |

---

## 📈 Model Performance & Evaluation

The Linear Regression model was evaluated on an unseen 20% holdout test dataset (311 samples) using **$R^2$ (Coefficient of Determination)**, **Root Mean Squared Error (RMSE)**, and **Mean Absolute Error (MAE)**:

| Metric | Training Set (80%) | Testing Holdout (20%) | Interpretation |
|---|---|---|---|
| **$R^2$ Score** | **0.9779** | **0.9811** | The model explains **98.11%** of variance in unseen test home prices |
| **RMSE** | **\$18,059.53** | **\$17,190.29** | Low standard deviation of prediction residuals on test properties |
| **MAE** | **\$14,570.59** | **\$13,551.59** | Mean absolute dollar error of predictions is under \$14k |
| **MAPE** | **2.15%** | **2.01%** | Average prediction error is only ~2% of property value |

---

## 🔍 Regression Coefficients & Economic Interpretation

The Ordinary Least Squares (OLS) model equation is expressed as:
$$\text{Price} = \beta_0 + \sum_{j=1}^{k} \beta_j X_j$$

* **Base Intercept ($\beta_0$):** **\$54,343.58** (Estimated foundational base land & zoning floor value).

### Detailed Feature Coefficients:

| Feature ($X_j$) | Coefficient ($\beta_j$) | Direction | Real-World Economic Meaning |
|---|---|---|---|
| `Square_Feet` | **+\$125.68** | Appreciating | Each additional square foot of living area adds **\$125.68** to property value. |
| `Overall_Quality` | **+\$22,994.82** | Appreciating | Each 1-point increase in build grade/finish quality increases value by **\$22,994.82**. |
| `Bathrooms` | **+\$23,931.14** | Appreciating | Each additional full/half bathroom adds **\$23,931.14** to property value. |
| `Bedrooms` | **+\$15,620.92** | Appreciating | Each additional bedroom increases estimated valuation by **\$15,620.92**. |
| `Garage_Cars` | **+\$13,289.33** | Appreciating | Each additional vehicle stall in garage adds **\$13,289.33**. |
| `Neighborhood_Score` | **+\$11,791.60** | Appreciating | Each 1-point boost in neighborhood desirability adds **\$11,791.60**. |
| `Lot_Area` | **+\$2.88** | Appreciating | Each additional square foot of parcel lot land adds **\$2.88**. |
| `House_Age` | **-\$709.31** | Depreciating | Every year elapsed since construction depreciates home value by **-\$709.31**. |

---

## 🔮 Example Inferences on Novel House Archetypes

Predictions generated using the saved model (`joblib.load('models/house_price_model.pkl')`) on four unseen house profiles:

| Profile Archetype | Living Area | Beds | Baths | Age | Garage | Lot Size | Quality | Desirability | **Predicted Price** |
|---|---|---|---|---|---|---|---|---|---|
| **1. Urban Starter Home** | 1,100 sqft | 2 | 1.0 | 20 yrs | 1 car | 4,500 sqft | 5/10 | 6.0/10 | **\$445,563.34** |
| **2. Suburban Family Residence** | 2,250 sqft | 4 | 2.5 | 12 yrs | 2 cars | 9,500 sqft | 7/10 | 7.5/10 | **\$754,287.70** |
| **3. Modern Luxury Villa** | 3,650 sqft | 5 | 4.0 | 2 yrs | 3 cars | 16,000 sqft | 9/10 | 9.2/10 | **\$1,086,911.56** |
| **4. Historic Vintage Cottage** | 1,600 sqft | 3 | 1.5 | 58 yrs | 1 car | 6,200 sqft | 6/10 | 6.8/10 | **\$546,365.03** |

---

## 📱 Interactive Web Application

The included Flask application provides:
1. **Real-time property valuation estimator** with live range validation.
2. **Instant economic breakdown** detailing the exact dollar contribution of each feature towards the final price.
3. **One-click presets** for starter homes, family residences, and luxury properties.
4. **Interactive model dashboard** displaying $R^2$, RMSE, OLS coefficient tables, and EDA plots.

To launch:
```bash
python app.py
```

---

## 💼 LinkedIn Internship Share Post Template (PDF Slide 4 Requirement)

Per the internship instructions on Slide 4, share your progress on LinkedIn:

> 🚀 **Excited to share my Machine Learning Internship Project @Syntecxhub!**  
>  
> For Week 1, I worked on **Project 1: House Price Prediction** using Linear Regression.  
>  
> 🔍 **Key Highlights:**  
> • Performed comprehensive data cleaning, missing value imputation, and feature engineering (House Age, Living-Lot ratio).  
> • Built and evaluated an Ordinary Least Squares (OLS) Linear Regression model achieving an **$R^2$ of 0.9811** and **RMSE of \$17,190** on test holdouts.  
> • Interpreted marginal feature coefficients (e.g., +\$125/sqft, +\$23k/quality score, -\$709/year age depreciation).  
> • Saved the production model and deployed an interactive Flask valuation web application with live visual analytics.  
>  
> 🔗 **GitHub Repository:** https://github.com/<your-username>/Syntecxhub_House_Price_Prediction  
>  
> A big thanks to **@Syntecxhub** for this hands-on learning opportunity!  
>  
> #Syntecxhub #MachineLearning #DataScience #Python #LinearRegression #AI #Internship

---

## 📝 Submission Checklist

- [x] Housing dataset loaded, cleaned (missing values, duplicates, outliers) & explored
- [x] Feature selection completed & 80/20 train/test split executed
- [x] Linear Regression model trained
- [x] Evaluated with RMSE and $R^2$, with full coefficient interpretation
- [x] Model saved (`.pkl`) and example predictions demonstrated
- [x] Interactive web application created with Flask
- [x] Interactive Jupyter Notebook included in `notebooks/`
- [x] Repository named `Syntecxhub_House_Price_Prediction`
- [x] LinkedIn update template prepared mentioning `@Syntecxhub`
