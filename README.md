# Interpretable Machine Learning for NYC Airbnb Price Prediction

*Samuel Orellana Mateo*  
*November 26th, 2024*

## Abstract

This project tackles the prediction of Airbnb rental prices in New York City using a dataset of 15,696 training points and 6,727 prediction instances, encompassing 51 diverse features such as property type, location, host information, and amenities. We begin with comprehensive exploratory data analysis to understand feature distributions, correlations, and data quality. Based on these insights, we perform data processing steps including one-hot and ordinal encoding, date transformations, and custom feature engineering (e.g., neighborhood mean pricing, amenities count, host responsiveness). Principal Component Analysis (PCA) reduces multicollinearity, while outlier handling stabilizes model training.

We employ Random Forest and XGBoost regression models, optimizing their hyperparameters through Bayesian optimization with Optuna. This automated tuning significantly enhances predictive accuracy, with our best XGBoost model achieving an RMSE of approximately 0.79 on the competition leaderboard, surpassing simpler models.

Beyond accuracy, we prioritize model interpretability by engineering intuitive composite features and conducting feature importance analyses. These reveal that room type, neighborhood price indicators, and listing capacity are key drivers of price variation. Our approach not only improves predictive performance but also provides clear insights into the factors influencing Airbnb prices in NYC.

---

*Samuel Orellana Mateo*  
samuel.orellanamateo@duke.edu
