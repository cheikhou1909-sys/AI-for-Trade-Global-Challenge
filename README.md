# AI for Trade Global Challenge — Team SABLE Repository

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains our solution for the **AI for Trade Global Challenge** organized by the Center for Collective Learning (CCL) with the Observatory of Economic Complexity (OEC). The goal is to forecast international trade flows for the United States and China for October 2025.

## 🎯 Challenge Overview

- **Objective**: Forecast trade flows (imports & exports) for USA and China in October 2025
- **Scope**: Top 20 trading partners trading ≥200 HS4 products
- **Evaluation**: sMAPE (symmetric Mean Absolute Percentage Error)
- **Submission Deadline**: October 31, 2025 (CET, midnight)

Full challenge details available in [`docs/challenge_details.md`](docs/challenge_details.md)

---

## 📊 Our Approach

### Data Sources

Our solution integrates multiple data sources:

1. **Trade Data**
   - OEC trade data for USA & China (2023-2024) — main training data
   - UN Comtrade monthly HS4 data (2021, 2022, 2025) — additional historical context
   
2. **External Economic Data**
   - **Exchange Rates**: Monthly USD rates from [Frankfurter API](https://www.frankfurter.app/)
   - **Commodity Prices**: Energy, metals, and agricultural commodities from [FRED API](https://fred.stlouisfed.org/)
   
3. **Economic Indicators**
   - **GDP & Macroeconomic Indicators**: World Bank data (constant prices, seasonally adjusted)
   - **REER**: Real Effective Exchange Rate from IMF
   
### Methodology

#  Full Mathematical Description of the Forecast Model

This document provides a **comprehensive mathematical explanation** of a forecasting model for **monthly trade values** by `(product × country × flow)`.

The model combines **naive baselines**, **seasonal and trend decomposition**, **lagged exogenous variables**, and a **robust Huber regression correction**, with **special handling for short or rare series**.

---

## 0. Notation

Let each series $s = (\text{product}, \text{country}, \text{flow})$ be observed monthly as:

$$
\{ (t_1, y_1), (t_2, y_2), ..., (t_T, y_T) \}, \quad y_t \ge 0
$$

We denote:
- $M_t \in \{1,\dots,12\}$ the **month** of $t$  
- $b_t$ the **base forecast**  
- $\hat y_t$ the **final prediction**  
- $x_t$ exogenous variables, possibly lagged

---

## 1. Base Forecast Candidates

For each series, several naive forecast candidates are computed:

1. **Last observed value**
$$
\text{last} = y_T
$$

2. **Same month last year**
$$
\text{seas12} = y_{T-12}, \quad \text{if available}
$$

3. **Moving averages**
$$
\text{ma3} = \frac{1}{3} \sum_{i=T-2}^{T} y_i, \quad
\text{ma6} = \frac{1}{6} \sum_{i=T-5}^{T} y_i
$$

4. **Drift extrapolation**
- Compute log-transformed slope over last 6 months:
$$
\beta = \frac{\sum_{i=T-5}^{T} (i-\bar i) (\log(1+y_i) - \overline{\log(1+y)} )}{\sum_{i=T-5}^{T} (i-\bar i)^2}
$$
- Extrapolated forecast:
$$
\text{drift_last} = y_T \cdot e^\beta
$$

5. **Seasonal-trend forecast** (see Section 2)

6. **Same-month median of last 3 years**
$$
\text{same_month_med3} = \text{median}\{y_t : M_t = M_T, t \in \text{last 3 years}\}
$$

**Base forecast** $b_{T+1}$ is the median of all valid candidates:

$$
b_{T+1} = \text{median}\{\text{last}, \text{seas12}, \text{ma3}, \text{ma6}, \text{drift_last}, \text{seasonal_ST}, \text{same_month_med3}\}
$$

---

## 2. Seasonal + Trend Decomposition

Assume a multiplicative model:

$$
y_t \approx g \cdot S_{M_t} \cdot \text{trend}_t
$$

### 2.1 Seasonal factor $S_m$

- Compute global scale $g$ as median of positive $y_t$ values:
$$
g = \text{median} \{ y_t : y_t > 0 \}
$$
- For each month $m$:
$$
S_m = \text{clip}\left( \frac{\text{median}\{y_t : M_t = m, y_t>0\}}{g}, 0.2, 5.0 \right)
$$

### 2.2 Deseasonalize and slope estimation

- Deseasonalized series:
$$
d_t = \frac{y_t}{S_{M_t}}
$$
- Slope on log-space (small window $w$):
$$
\beta = \frac{\sum_{i=T-w+1}^{T} (i-\bar i) (\log(1 + d_i) - \overline{\log(1+d)})}{\sum_{i=T-w+1}^{T} (i-\bar i)^2}
$$

### 2.3 Forecast next month

$$
\hat d_{T+1} = d_T \cdot e^\beta, \quad
\text{seasonal_ST} = \hat d_{T+1} \cdot S_{M_{T+1}}
$$

- Blend with same-month median and last value:
$$
\text{seasonal_ST} = \text{median}\{\text{seasonal_ST}, y_T, \text{same_month_med3}\}
$$

---

## 3. Exogenous Variable Selection

For candidate numeric exogenous variables $x_j$:

1. For each lag $L \in \{1,3,6\}$:
   - Form pairs $(x_{t-L}, y_t)$ over all series
2. Compute Spearman correlation $\rho_{x_j, L}$
3. Keep top `$k$` pairs $(x,L)$ with highest $|\rho_{x,L}|$

These features are used in the Huber regression.

---

## 4. Feature Construction

For each series and target month:

| Feature | Formula / Description |
|:--|:--|
| Month | $M_{T+1}$ |
| Binary month indicators | `is_jan`, `is_dec` |
| Encodings | Mean target per product/country/flow: $\text{enc\_prod} = \text{mean}(y)$ for that product |
| Seasonality | $S_{M_{T+1}}$ |
| Lags | $y_T$, $y_{T-1}$, $y_{T-2}$, $y_{T-3}$, $y_{T-6}$, $y_{T-12}$ |
| Exogenous | Selected top-$k$ lagged $x_j$ |
| Optional: log slope | $\beta$ from deseasonalized last $w$ points |

---

## 5. Robust Correction: Huber Regression

### 5.1 Target transformation

Define relative correction in log-space:

$$
z_t = \log(1 + y_t) - \log(1 + b_t)
$$

- $b_t$ is base forecast
- $z_t$ is the **residual correction**

### 5.2 Model training

- Stack all series and backtests
- Train **HuberRegressor** to predict $z_t$ from features $X_t$:

$$
\hat z_{T+1} = f_{\text{Huber}}(X_{T+1})
$$

Huber loss:
$$
L_\delta(r) =
\begin{cases}
\frac{1}{2} r^2 & |r| \le \delta \\
\delta (|r| - \frac{1}{2}\delta) & |r| > \delta
\end{cases}, \quad r = z_t - \hat z_t
$$

---

## 6. Final Forecast

- Log-space corrected forecast:
$$
\hat y_{\text{corr}} = \exp(\log(1+b_{T+1}) + \hat z_{T+1}) - 1
$$

- For rare series (mostly zeros):
$$
y_{\text{corr}} = \max(y_{\text{floor}}, \min(\hat y_{\text{corr}}, y_{T} \cdot g_{\text{cap}}))
$$

- Weighted blend with base forecast:
$$
\hat y_{T+1} = w \cdot y_{\text{corr}} + (1-w) \cdot b_{T+1}, \quad w = 0.6
$$

---

## 7. Rare Series Handling

A series is considered rare if fraction of nonzero $y_t$:

$$
\frac{\#\{y_t>0\}}{T} < f_{\text{rare}}
$$

- Floor applied:
$$
y_{\text{floor}} = \text{median of last nonzero k values} \cdot f_{\text{floor}}
$$
- Cap growth:
$$
y_{\text{corr}} \le y_T \cdot g_{\text{cap}}
$$

---

## 8. Evaluation: Micro sMAPE

Global metric:
$$
\text{sMAPE} = 100 \cdot \frac{1}{N} \sum_{i=1}^{N} \frac{2 |y_i - \hat y_i|}{|y_i| + |\hat y_i| + \varepsilon}
$$

Top-20 metric: only top 20 countries per `(product × flow)` by **sum of trade_value**.

---

## 9. Summary of Model Flow

1. **Split train/test**  
2. **Identify series** `(product × country × flow)`  
3. **Compute naive candidates** (last, MA, drift, seasonal, same-month median)  
4. **Compute base forecast** $b$ = median(candidates)  
5. **Compute seasonal-trend factors** $S_m$ and slope $\beta$  
6. **Select exogenous features** by Spearman correlation  
7. **Construct features**: lags, seasonality, encodings, exog  
8. **Backtest and fit Huber regression** on $\log$ correction  
9. **Predict next month**: apply Huber correction in log-space  
10. **Rare series adjustment**: floor & growth cap  
11. **Blend with base forecast** for final prediction  
12. **Evaluate with sMAPE** (global and top-20)

---

This approach ensures **robustness, interpretability, and high accuracy**, combining **time series heuristics** with **machine learning residual correction**.


---

## 🗂️ Repository Structure

```
AI-for-Trade-Global-Challenge/
│
├── README.md                          # This file - project overview
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── Makefile                          # Common commands
│
├── configs/
│   └── config.yaml                   # Centralized configuration
│
├── docs/
│   ├── challenge_details.md          # Full challenge description
│   └── data_pipeline.md              # Data processing documentation
│
├── inputs/                           # Input data (not versioned)
│   ├── raw/                          # Raw provided data
│   │   ├── ForParticipants/          # OEC training data (2023-2024)
│   │   └── comtrade_monthly_hs4_outputs/  # Comtrade data (2021,2022,2025)
│   ├── external/                     # External data sources
│   └── reference/                    # Lookups, code lists
│       ├── code_hs4.xlsx             # HS4 product code mappings
│       ├── df_long.csv               # Economic indicators (World Bank)
│       └── EER_COUNTRIES.csv         # REER data (IMF)
│
├── outputs/                          # Generated artifacts (not versioned)
│   ├── interim/                      # Intermediate processing results
│   │   ├── exchange_rates.csv        # Fetched exchange rates
│   │   ├── commodity_prices.csv      # Fetched commodity prices
│   │   └── comtrade_merged/          # Merged Comtrade files
│   ├── processed/                    # Final processed datasets
│   │   ├── USA_2023_finale.csv       # Processed USA 2023 data
│   │   ├── USA_2024_finale.csv       # Processed USA 2024 data
│   │   ├── china_2023_finale.csv     # Processed China 2023 data
│   │   ├── china_2024_finale.csv     # Processed China 2024 data
│   │   ├── with_external/            # Data with external variables
│   │   ├── with_indicators/          # Data with economic indicators
│   │   └── comtrade_final/           # Processed Comtrade data
│   ├── forecasts/                    # Final forecast CSVs
│   ├── evaluation/                   # Validation reports and scores
│   └── reports/                      # Figures, analysis, writeups
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── pipeline.py                   # Main data processing pipeline
│   ├── data_processing/              # Data processing modules
│   │   ├── __init__.py
│   │   ├── process_trade_data.py     # Process 2023-2024 trade data
│   │   ├── external_data.py          # Fetch exchange rates & commodities
│   │   ├── comtrade_processor.py     # Process Comtrade data
│   │   └── indicators.py             # Add GDP, REER indicators
│   ├── metrics.py                    # sMAPE and evaluation metrics
│   ├── validate_submission.py        # Validate submission format
│   ├── evaluate.py                   # Compute sMAPE scores
│   └── forecast.py                   # Forecasting models
│
├── scripts/                          # Convenience scripts
│   ├── run_pipeline.sh               # Run complete data pipeline
│   ├── make_submission.sh            # Generate forecast submission
│   ├── validate_submission.sh        # Validate submission file
│   └── evaluate_submission.sh        # Evaluate against ground truth
│
├── notebooks/                        # Jupyter notebooks
│   ├── README.md                     # Notebook guidance
│   ├── SABLE_model_training.ipynb    # Main model training notebook
│   └── comtrade_data_processing.ipynb # Comtrade data processing
│
├── submissions/
│   ├── README.md                     # Submission guidelines
│   └── template_submission.csv       # Example submission format
│
├── tests/
│   └── test_metrics.py               # Unit tests
│
└── [Legacy files - to be deprecated]
    ├── 1_data_processing.py          # → moved to src/data_processing/
    └── 2_rajout_donnees_externes.py  # → moved to src/data_processing/
```

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/AI-for-Trade-Global-Challenge.git
cd AI-for-Trade-Global-Challenge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### 2. Prepare Data

Download the required datasets and organize them:

```
inputs/
  raw/
    ForParticipants/
      trade_s_usa_state_m_hs_2023.csv
      trade_s_usa_state_m_hs_2024.csv
      trade_s_chn_m_hs_2023.csv
      trade_s_chn_m_hs_2024.csv
    comtrade_monthly_hs4_outputs/
      [Comtrade monthly files]
  reference/
    code_hs4.xlsx
    df_long.csv
    EER_COUNTRIES.csv
```

### 3. Run Data Processing Pipeline

Set your FRED API key (optional, for commodity prices):
```bash
export FRED_API_KEY="your_api_key_here"
```

Run the complete pipeline:
```bash
python src/pipeline.py --fred-api-key $FRED_API_KEY
```

Or run specific steps:
```bash
# Skip external data fetching
python src/pipeline.py --skip-external

# Skip Comtrade processing
python src/pipeline.py --skip-comtrade

# Skip adding indicators
python src/pipeline.py --skip-indicators
```

### 4. Train Models & Generate Forecast

Open the main modeling notebook:
```bash
jupyter notebook notebooks/SABLE_model_training.ipynb
```

Or use the forecast script:
```bash
python src/forecast.py --output outputs/forecasts/submission.csv
```

### 5. Validate & Evaluate

```bash
# Validate submission format
bash scripts/validate_submission.sh outputs/forecasts/submission.csv

# Evaluate against ground truth (when available)
bash scripts/evaluate_submission.sh \
  outputs/forecasts/submission.csv \
  inputs/reference/oct2025_truth.csv
```

---

## 📋 Data Processing Pipeline Details

### Step 1: Process Main Trade Data (2023-2024)

Processes OEC trade data for USA and China:
- Aggregate product IDs to HS4 level (first 4 digits)
- Calculate number of unique products per country-month
- Filter countries with ≥200 products (challenge requirement)
- Add product names from HS4 code mappings

**Module**: `src/data_processing/process_trade_data.py`

### Step 2: Fetch External Data

Retrieves economic data from external APIs:
- **Exchange Rates**: Monthly USD rates for all currencies (Frankfurter API)
- **Commodity Prices**: Energy, metals, agriculture prices (FRED API)

**Module**: `src/data_processing/external_data.py`

### Step 3: Process Comtrade Data (2021, 2022, 2025)

Processes additional UN Comtrade datasets:
- Merge monthly files by year and trade flow
- Handle encoding issues and missing columns
- Normalize to standard format
- Filter by quantity thresholds

**Module**: `src/data_processing/comtrade_processor.py`

### Step 4: Add Economic Indicators

Integrates macroeconomic indicators:
- **GDP Components**: Final consumption, capital formation, inventories
- **REER**: Real Effective Exchange Rate (IMF data)
- Merge with trade data by country and month

**Module**: `src/data_processing/indicators.py`

---

## 🔧 Configuration

All paths and parameters are centralized in `configs/config.yaml`:

- **Paths**: Input/output directories
- **Data Processing**: Minimum products threshold, date ranges, file names
- **Model**: Feature categories, target variable, validation split
- **Submission**: Required columns, validation rules

Edit this file to customize the pipeline behavior.

---

## 📈 Model Training

Our modeling approach uses ensemble methods:

1. **Feature Engineering**
   - Temporal features: month trends, seasonality
   - Categorical encoding: target encoding for countries/products
   - Economic indicators: normalized and lagged features
   - Interaction features: country×product, country×month

2. **Model Architecture**
   - Primary: CatBoost (handles categorical features natively)
   - Ensemble: XGBoost, LightGBM, RandomForest
   - Stacking: Meta-learner combines predictions

3. **Validation Strategy**
   - Time-based split: Last 6 months for validation
   - Cross-validation: By country and product groups
   - Metric: sMAPE (official competition metric)

See `notebooks/SABLE_model_training.ipynb` for full details.

---

## 📝 Submission Format

Submissions must be CSV files with columns:
```
"Country1","Country2","ProductCode","TradeFlow","Value"
```

Example:
```csv
"USA","CHL","8404","Export","1234567"
"USA","CHN","8405","Import","9876543"
"CHN","USA","8404","Export","5555555"
```

- **Country1, Country2**: ISO-3 codes (e.g., USA, CHN, DEU)
- **ProductCode**: 4-digit HS4 code
- **TradeFlow**: "Export" or "Import"
- **Value**: Trade value in USD (non-negative)

Use `src/validate_submission.py` to check format before submitting.

---

## 🧪 Testing

Run unit tests:
```bash
python -m pytest tests/
```

Test specific module:
```bash
python -m pytest tests/test_metrics.py
```

---

## 📚 Documentation

- **Challenge Details**: [`docs/challenge_details.md`](docs/challenge_details.md)
- **Data Pipeline**: Documentation of each processing step
- **Notebooks**: Analysis and modeling documentation in Jupyter notebooks
- **API Reference**: Docstrings in all Python modules

---

## 🤝 Team & Contribution

**Team SABLE** — AI4Trade Global Challenge 2025

### Project Structure Notes

- **Inputs**: Not version controlled (add to `.gitignore`). Keep data local.
- **Outputs**: Generated artifacts, not version controlled.
- **Legacy Files**: `1_data_processing.py` and `2_rajout_donnees_externes.py` are legacy scripts. Use the modular `src/data_processing/` modules instead.

### Development Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OEC** (Observatory of Economic Complexity) for trade data
- **CCL** (Center for Collective Learning) for organizing the challenge
- **World Bank** for economic indicators
- **IMF** for REER data
- **FRED** for commodity price data

---

## 📞 Contact & Links

- **Challenge Website**: [Link to challenge page]
- **OEC Platform**: https://oec.world/
- **Documentation**: See `docs/` folder
- **Issues**: Use GitHub Issues for questions/bugs

---

**Good luck with the challenge! 🚀**
