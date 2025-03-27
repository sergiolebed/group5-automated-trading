# group5-automated-trading
# Automated Daily Trading System

This repository contains all files and documentation for the Group 5 project (Section 2, MBD): an automated daily trading system focusing on airline stocks.

### Overview

This project integrates data extraction, transformation, machine learning modeling, and a Streamlit web-based visualization.

### Key Functionalities

- **Data Extraction**: Utilizes the SimFin API for financial data (share prices, income statements, balance sheets).
- **Data Transformation (ETL)**: Includes cleaning, normalization (MinMaxScaler), and merging processes.
- **Machine Learning Model**: Predicts next-day stock movements using RSI, moving averages, Bollinger Bands, and trading volume.
- **Web Application**: Interactive visualization for historical/real-time data, ML trading signals, and airline stock comparisons.

### Repository Structure

```
Automated-Daily-Trading-System/
├── src/
│   ├── data/               # Data loading and ETL scripts
│   ├── models/             # ML model scripts
│   ├── web_app/            # Streamlit web app files
│   └── utils/              # Utility functions
├── requirements.txt
└── README.md
```

### How to Run

1. Clone the repository:

```bash
git clone https://github.com/username/group5-automated-trading.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Streamlit App:

```bash
streamlit run src/web_app/app.py
```

### Team Members

- Anastasia Chappel
- Felix Goossens
- Marta Pérez
- Sergio Lebed
- Youseff Hakim





