# 🎬 Cinema Audience Forecasting

> A hybrid machine learning system that combines rule-based forecasting with ensemble ML models to predict daily cinema audience counts.

---

## 📋 Overview

This project predicts daily cinema audience attendance using historical visit patterns, booking data, and calendar information. The solution implements a **smart two-stage approach** that automatically selects and tunes the best-performing model.

### Key Innovation
Instead of manually trying different models, the system:
1. Trains 5 models with default parameters (fast screening)
2. Identifies the best performer automatically
3. Deep-tunes only the winning model (saves time)
4. Uses the optimized model for predictions

---

## ✨ Features

- **Hybrid Approach**: Rule-based baseline + ML learning
- **Smart Model Selection**: Automatic identification of best model
- **Efficient Tuning**: Only tunes the best model (60-70% faster)
- **Visual Performance Reports**: Charts comparing all models
- **Time Series Features**: Lags, rolling averages, EMA, differences
- **Robust Preprocessing**: Scaling, imputation, encoding

---

## 🚀 Quick Start

### 1️⃣ Setup

**Requirements:**
- Python 3.8+
- Kaggle account

**Install Dependencies:**
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

### 2️⃣ Run on Kaggle

1. Upload the notebook to [Kaggle](https://www.kaggle.com/)
2. Add the competition dataset
3. Run all cells
4. Download `submission.csv` 

### 3️⃣ Expected Output

```
📊 Training records: 2,14,046

Step 1: Quick screening with default parameters...
✓ All models trained

Best Model (before tuning): Xgboost and Random Forest
   R² Score: 0.55

Step 2: Deep tuning of Xgboost and Random Forest...
✅ Tuning Complete!
   Best Parameters: {'n_estimators': 180, 'max_depth': 5, ...}

📈 FINAL MODEL PERFORMANCE COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model                   R² Score    MAE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost                 0.559      118.20 ⭐ TUNED
LightGBM                0.555      115.30 
Random Forest           0.557      119.80 ⭐ TUNED
Ridge                   0.483      123.40
Gradient Boost          0.525      120.10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ submission.csv generated successfully!
```

---

## 🔬 Methodology

### Architecture

```
┌─────────────────────────────────────────────────────┐
│           INPUT DATA                                 │
│  • Visit History  • Bookings  • Calendar Info       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│       FEATURE ENGINEERING                           │
│  • Time series lags (1, 7, 14 days)                 │
│  • Rolling averages (7, 14, 30 days)                │
│  • Exponential moving average                        │
│  • Theater encoding & frequency                      │
│  • Holiday/Weekend flags                             │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│       RULE-BASED BASELINE                            │
│  Weighted historical patterns → Baseline Prediction  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│       MODEL SCREENING (Fast)                         │
│  XGBoost │ LightGBM │ RF │ Ridge │ GB               │
│  Train all with defaults → Find best                 │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│       HYPERPARAMETER TUNING                          │
│  RandomizedSearchCV on best model only               │
│  20 iterations × 3-fold CV                           │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│       FINAL PREDICTION                               │
│  Baseline + ML Residual Correction                   │
└─────────────────────────────────────────────────────┘
```

### Rule-Based Formula

```python
Prediction = 0.70 × theater_baseline +
             0.07 × rolling_mean_7d +
             0.05 × rolling_mean_14d +
             0.07 × lag_1 +
             0.38 × tickets_booked +
             0.04 × ema_7d +
             (holiday/weekend adjustments)
```

### ML Models Evaluated

| Model | Description | Hyperparameters Tuned |
|-------|-------------|----------------------|
| **XGBoost** | Gradient boosting with regularization | n_estimators, max_depth, learning_rate, subsample |
| **LightGBM** | Fast gradient boosting | n_estimators, max_depth, num_leaves, learning_rate |
| **Random Forest** | Ensemble of decision trees | n_estimators, max_depth, min_samples_split |
| **Ridge** | Linear regression with L2 penalty | alpha |
| **Gradient Boosting** | Classic sklearn GB | n_estimators, max_depth, learning_rate |

---

## 📊 Results

### Performance Metrics

- **Rule-based Baseline**: R² = 0.41, MAE = 125.5
- **Best Model (Tuned)**: R² = 0.45+, MAE = 115-120
- **Improvement**: ~10% better than baseline

### Visual Outputs

The notebook generates:
1. **Model Performance Comparison Chart** - Side-by-side R² and MAE comparison
2. **Dataset Exploration Dashboard** - 6-panel overview of data structure

---

## 🛠️ Technical Details

### Data Preprocessing

```python
✓ Missing Values: Median imputation
✓ Scaling: RobustScaler (outlier-resistant)
✓ Encoding: Theater frequency & baseline encoding
✓ Time Series: Proper train-test split (no data leakage)
```

### Feature Categories

| Category | Features |
|----------|----------|
| **Temporal** | day_of_week, month, weekend_flag, holiday_flag |
| **Lag Features** | audience_lag1, audience_lag7, audience_lag14 |
| **Rolling Stats** | rolling_mean_7d, rolling_mean_14d, ema_7d |
| **Business** | tickets_booked, theater_baseline |
| **Derived** | diff_1d, diff_7d, theater_frequency |

### Why This Approach Works

1. **Rule-based captures domain knowledge** (theater patterns, seasonality)
2. **ML learns complex residuals** (what rules miss)
3. **Smart tuning saves time** (60-70% faster than tuning all models)
4. **Robust preprocessing** prevents overfitting

---

## 📈 Model Performance Comparison

```
Performance Improvement Over Baseline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost:          +6.1%
LightGBM:         +10.2% 
Random Forest:    +4.4%
Ridge:            +1.2%
Gradient Boost:   +5.1%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📁 Files in Repository

```
cinema-audience-forecasting/
│
├── cinema_forecasting_v4.5.ipynb    # Main notebook (all code)
├── README.md                         # This file
├── requirements.txt                  # Dependencies
└── outputs/
    ├── submission.csv                       # Final predictions
    └── model_performance_comparison.png     # Performance charts
```

---

## 💡 Key Learnings

### What Worked Well
- ✅ Hybrid approach (rule + ML) beats pure ML
- ✅ Theater-level encoding captures venue differences
- ✅ Booking data is highly predictive
- ✅ Tuning only best model saves significant time

### Challenges Faced
- ⚠️ Missing values in lag features (solved with median imputation)
- ⚠️ Outliers in audience counts (solved with RobustScaler)
- ⚠️ Time series data leakage prevention (solved with proper CV)

---

## 🔮 Future Improvements

- [ ] Add external features (weather, movie ratings, genres)
- [ ] Implement LSTM/GRU for sequential patterns
- [ ] Add movie-level predictions (not just theater-level)
- [ ] Create separate models for weekdays vs weekends
- [ ] Experiment with stacking/blending multiple models
- [ ] Add anomaly detection for unusual patterns

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Shreya Saxena**
- GitHub: [@yourusername](https://github.com/shreyasaxena21)
- Kaggle: [@yourkaggleusername](https://www.kaggle.com/shreya22f3001013)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/shreya-saxena-16a011246)

---

## 🙏 Acknowledgments

- Dataset provided by [Cinema_Audience_Forecasting_challenge] on Kaggle
- Inspired by hybrid forecasting approaches in retail and entertainment
- Thanks to the ML community for open-source libraries

---

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn Time Series Guide](https://scikit-learn.org/stable/)
- [Kaggle Learn: Time Series](https://www.kaggle.com/learn/time-series)

---

<div align="center">

**⭐ If you found this helpful, please star the repository! ⭐**

</div>
