# FX Investment Strategies

Empirical analysis of carry, momentum, and dollar FX strategies using monthly spot/forward rates from WRDS and short-term interest rates from FRED. Covers Q1–Q8 of the analysis: excess return construction, strategy performance, in-sample portfolio optimization, and out-of-sample EWMA/Fama-MacBeth CMVE portfolios.

**Sample period:** Jan 1991 – Dec 2024  
**Evaluation period:** Jan 1995 – Dec 2024  
**Currencies:** AUD, CAD, EUR, GBP, JPY, NZD (vs. USD)

---

## Project Structure

```
investments-project/
├── main.py                        # Single entry point — runs Q1–Q8
├── config.py                      # WRDS username
├── requirements.txt
│
├── src/
│   ├── core/
│   │   └── excess_return.py       # ExcessReturnCalculator (Q1–Q4)
│   ├── strategies/
│   │   ├── momentum.py            # CS-MOM and TS-MOM (Q5)
│   │   └── dollar.py              # DOLLAR and DOLLAR-CARRY (Q6)
│   ├── portfolios/
│   │   ├── construction.py        # EW, risk-parity, mean-variance (Q7)
│   │   └── oos.py                 # EWMA covariance, FM regressions, CCV portfolios (Q8)
│   └── fetchers/
│       ├── fx_data_fetcher.py     # WRDS FX spot/forward data
│       └── interest_rate_fetcher.py  # FRED short-term interest rates
│
├── scripts/                       # Run individual questions in isolation
│   ├── fetch_data.py              # Data fetch only (requires WRDS access)
│   ├── run_q1_q4.py
│   ├── run_q5_q6.py
│   ├── run_q7.py
│   └── run_q8.py
│
├── data/
│   ├── raw/                       # CSV files written by fetchers (git-ignored)
│   └── output/                    # Tables, weights, portfolio returns (git-ignored)
│
└── notebooks/
    └── tests.ipynb
```

---

## Setup

```bash
pip install -r requirements.txt
```

You need a [WRDS](https://wrds-www.wharton.upenn.edu/) account. Set your username in [config.py](config.py):

```python
WRDS_USERNAME = "your_wrds_username"
```

---

## Running

### Full pipeline (recommended)

```bash
python main.py
```

`main.py` automatically fetches raw data from WRDS/FRED if `data/raw/` is empty, then runs Q1–Q8 sequentially and prints all tables to stdout. Output files are saved to `data/output/`.

### Fetch data separately

```bash
python scripts/fetch_data.py
```

### Run individual questions

```bash
python scripts/run_q1_q4.py   # Table 1, carry weights, correlations
python scripts/run_q5_q6.py   # Momentum and dollar strategies
python scripts/run_q7.py      # In-sample portfolio analysis (Table 3)
python scripts/run_q8.py      # Out-of-sample CMVE portfolios (Table 4)
```

All scripts must be run from the project root directory.

---

## Analysis Overview

| Question | Description | Key output |
|----------|-------------|------------|
| Q1 | Monthly excess returns per currency | Table 1: mean, std, Sharpe |
| Q2 | CS-CARRY strategy (rank on interest rate differential) | Table 2 |
| Q3 | TS-CARRY strategy (time-series carry signal) | Table 2 |
| Q4 | Carry weight evolution plot + forward discount correlations | `data/output/cs_carry_weights.png` |
| Q5 | CS-MOM and TS-MOM momentum strategies; regression of TS-CARRY on momentum | Table 2 |
| Q6 | DOLLAR and DOLLAR-CARRY strategies; regression of TS-CARRY on momentum + dollar | Table 2 |
| Q7 | In-sample EW, risk-parity, and mean-variance portfolios across all factors | Table 3 |
| Q8 | Out-of-sample CCV-Exp (expanding mean + EWMA Σ) and CCV-FM (Fama-MacBeth μ + EWMA Σ) | Table 4 |

---

## Outputs

| File | Contents |
|------|----------|
| `data/output/table3_question7.csv` | In-sample portfolio stats |
| `data/output/table4_question8.csv` | Out-of-sample portfolio stats |
| `data/output/question7_portfolio_returns.csv` | Monthly portfolio returns (Q7) |
| `data/output/question8_ccv_returns.csv` | Monthly CCV-Exp / CCV-FM returns |
| `data/output/question8_fm_estimates.csv` | Period-by-period FM coefficients |
| `data/output/cs_carry_weights.png` | Carry strategy weight evolution |
