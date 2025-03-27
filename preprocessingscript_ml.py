import os
from dotenv import load_dotenv
import simfin as sf
from fuzzywuzzy import process
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Load environment variables and set SimFin API key and data directory
load_dotenv()
api_token = os.getenv('API_KEY')

# --------------------------------------------------
# Data Loading and Company Data Access Classes
# --------------------------------------------------

class DataLoader:
    def __init__(self, api_token: str, data_dir: str = '~/simfin_data/'):
        self.api_token = api_token
        self.data_dir = data_dir
        sf.set_api_key(self.api_token)
        sf.set_data_dir(self.data_dir)

    def load_share_prices(self) -> pd.DataFrame:
        df_share_price = sf.load_shareprices(market='us', variant='daily')
        df_share_price.reset_index(inplace=True)
        return df_share_price

    def load_companies(self) -> pd.DataFrame:
        df_companies = sf.load_companies(market='us')
        df_companies.reset_index(inplace=True)
        return df_companies

    def load_income(self) -> pd.DataFrame:
        df_income = sf.load_income(variant='quarterly', market='us')
        df_income.reset_index(inplace=True)
        return df_income

    def load_balance_sheet(self) -> pd.DataFrame:
        df_balance_sheet = sf.load_balance(variant='quarterly', market='us')
        df_balance_sheet.reset_index(inplace=True)
        return df_balance_sheet


class AccessCompanyData:
    def __init__(self, company_name: str, loader: DataLoader):
        self.loader = loader
        self.ticker = self.find_ticker(company_name)

    def find_ticker(self, company_name: str) -> str:
        df = self.loader.load_companies()
        best_match = process.extractOne(company_name, df['Company Name'])
        if best_match:
            matched_name = best_match[0]
            return df.loc[df['Company Name'] == matched_name, 'Ticker'].values[0]
        return None

    def company_info(self) -> pd.Series:
        df = self.loader.load_companies()
        if self.ticker:
            return df[df['Ticker'] == self.ticker].iloc[0]
        return None

    def company_share_price(self) -> pd.DataFrame:
        df = self.loader.load_share_prices()
        if self.ticker:
            sp = df[df['Ticker'] == self.ticker]
            to_drop = ['SimFinId', 'Adj. Close', 'Dividend']
            sp = sp.drop(columns=[c for c in to_drop if c in sp.columns])
            return sp
        return None

    def company_balance_sheet(self) -> pd.DataFrame:
        df = self.loader.load_balance_sheet()
        if self.ticker:
            bs = df[df['Ticker'] == self.ticker]
            to_drop = [
                'Ticker', 'SimFinId', 'Currency', 'Fiscal Year', 'Fiscal Period',
                'Publish Date', 'Restated Date', 'Shares (Basic)', 'Shares (Diluted)',
                'Treasury Stock', 'Long Term Investments & Receivables',
                'Other Long Term Assets', 'Share Capital & Additional Paid-In Capital',
                'Retained Earnings'
            ]
            bs = bs.drop(columns=[c for c in to_drop if c in bs.columns])
            return bs
        return None

    def company_income_statement(self) -> pd.DataFrame:
        df = self.loader.load_income()
        if self.ticker:
            inc = df[df['Ticker'] == self.ticker]
            to_drop = [
                'Ticker', 'SimFinId', 'Currency', 'Fiscal Year', 'Fiscal Period',
                'Publish Date', 'Restated Date', 'Shares (Basic)', 'Shares (Diluted)',
                'Abnormal Gains (Losses)', 'Net Extraordinary Gains (Losses)',
                'Research & Development'
            ]
            inc = inc.drop(columns=[c for c in to_drop if c in inc.columns])
            return inc
        return None

# --------------------------------------------------
# Data Cleaning, Feature Engineering, and Enhancement
# --------------------------------------------------

class CleanCompanyData:
    def __init__(self, share_price: pd.DataFrame, balance_sheet: pd.DataFrame, income: pd.DataFrame):
        self.share_price = share_price.copy()
        self.balance_sheet = balance_sheet.copy()
        self.income = income.copy()
        self.cleaned_df = self._process_data()

    def _process_data(self) -> pd.DataFrame:
        merged = self._align_by_report_date(self.share_price, self.balance_sheet, 'Date', 'Report Date')
        final = self._align_by_report_date(merged, self.income, 'Date', 'Report Date')
        if 'Report Date' in final.columns:
            final.drop(columns='Report Date', inplace=True)
        return final.dropna()

    def _align_by_report_date(self, df_base: pd.DataFrame, df_quarterly: pd.DataFrame,
                              date_col: str, report_date_col: str) -> pd.DataFrame:
        df_base = df_base.copy()
        df_quarterly = df_quarterly.copy()
        df_base[date_col] = pd.to_datetime(df_base[date_col])
        df_quarterly[report_date_col] = pd.to_datetime(df_quarterly[report_date_col])
        df_quarterly.sort_values(by=report_date_col, inplace=True)
        df_base['Report Date'] = df_base[date_col].apply(
            lambda d: df_quarterly.loc[df_quarterly[report_date_col] <= d, report_date_col].max()
            if not df_quarterly.loc[df_quarterly[report_date_col] <= d].empty else pd.NaT
        )
        return df_base.merge(df_quarterly, on='Report Date', how='left')

    def get_enhanced_data(self) -> pd.DataFrame:
        enhancer = EnhancingFeatures(self.cleaned_df)
        enhancer.daily_return()
        enhancer.macd()
        enhancer.rsi()
        enhancer.lag_one()
        enhancer.bollinger_bands()
        enhancer.rolling_volatility()
        enhancer.normalized_trading_volume()
        enhancer.shifted_return()
        enhancer.trend()
        enhanced_df = enhancer.final_clean()
        return enhanced_df


class EnhancingFeatures:
    def __init__(self, cleaned_df: pd.DataFrame):
        self.cleaned_df = cleaned_df.copy()

    def daily_return(self) -> None:
        self.cleaned_df['Daily Return'] = (
            (self.cleaned_df['Close'] - self.cleaned_df['Close'].shift(1))
            / self.cleaned_df['Close'].shift(1)
        )

    def macd(self) -> None:
        self.cleaned_df['EMA_12'] = self.cleaned_df['Close'].ewm(span=12, adjust=False).mean()
        self.cleaned_df['EMA_26'] = self.cleaned_df['Close'].ewm(span=26, adjust=False).mean()
        self.cleaned_df['MACD'] = self.cleaned_df['EMA_12'] - self.cleaned_df['EMA_26']

    def rsi(self) -> None:
        delta = self.cleaned_df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.cleaned_df['RSI'] = 100 - (100 / (1 + rs))

    def lag_one(self) -> None:
        self.cleaned_df['MACD_Lag1'] = self.cleaned_df['MACD'].shift(1)
        self.cleaned_df['RSI_Lag1'] = self.cleaned_df['RSI'].shift(1)

    def bollinger_bands(self) -> None:
        self.cleaned_df['SMA_20'] = self.cleaned_df['Close'].rolling(window=20).mean()
        self.cleaned_df['Rolling_STD_20'] = self.cleaned_df['Close'].rolling(window=20).std()

    def rolling_volatility(self, window: int = 20) -> None:
        self.cleaned_df[f'Rolling_Volatility_{window}'] = self.cleaned_df['Daily Return'].rolling(window=window).std()

    def normalized_trading_volume(self) -> None:
        self.cleaned_df['Normalized_Volume'] = (
            self.cleaned_df['Volume'] / self.cleaned_df['Volume'].rolling(window=30).mean()
        )

    def shifted_return(self, shift_period: int = 1) -> None:
        self.cleaned_df[f'Return_{shift_period}D'] = self.cleaned_df['Daily Return'].shift(shift_period)

    def trend(self) -> None:
        self.cleaned_df['Trend'] = self.cleaned_df['Close'] - self.cleaned_df['Close'].shift(10)

    def final_clean(self) -> pd.DataFrame:
        self.cleaned_df.fillna(method='bfill', inplace=True)
        return self.cleaned_df

# --------------------------------------------------
# Additional Utilities: Graphs and Normalization
# --------------------------------------------------

class Graphs:
    def __init__(self, final_df: pd.DataFrame):
        self.final_df = final_df.copy()

    def close_sma(self):
        return self.final_df[['Close', 'SMA_20']].plot(figsize=(12, 6), title="Close Price and SMA_20")

    def close_macd(self):
        return self.final_df[['MACD', 'Close']].plot(figsize=(12, 6), title="Close Price and MACD")

    def rsi(self):
        return self.final_df['RSI'].plot(figsize=(12, 6), title="Relative Strength Index")


class Normalization:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def min_max_scale(self, column: str) -> None:
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)

# --------------------------------------------------
# Market Aggregation and Machine Learning
# --------------------------------------------------

class MarketAggregator:
    def __init__(self, combined_df: pd.DataFrame):
        self.combined_df = combined_df.copy()
        self.combined_df['Date'] = pd.to_datetime(self.combined_df['Date'])
        # Drop 'Ticker' if present
        if 'Ticker' in self.combined_df.columns:
            self.combined_df.drop(columns=['Ticker'], inplace=True)

        # Aggregation: using 'mean' for OHLC, sum for Volume, etc.
        self.aggregation_funcs = {
            'Open': 'mean',
            'High': 'mean',
            'Low': 'mean',
            'Close': 'mean',
            'Volume': 'sum',
            'Daily Return': 'mean',
            'EMA_12': 'mean',
            'EMA_26': 'mean',
            'MACD': 'mean',
            'RSI': 'mean',
            'MACD_Lag1': 'mean',
            'RSI_Lag1': 'mean',
            'SMA_20': 'mean',
            'Rolling_STD_20': 'mean',
            'Rolling_Volatility_20': 'mean',
            'Normalized_Volume': 'mean',
            'Return_1D': 'mean',
            'Trend': 'mean',
        }

    def aggregate(self) -> pd.DataFrame:
        market_df = self.combined_df.groupby('Date').agg(self.aggregation_funcs).reset_index()
        return market_df


class MachineLearning:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # If Ticker is still present, drop it
        if 'Ticker' in self.df.columns:
            self.df.drop(columns=['Ticker'], inplace=True)
        self.df = self.add_target_column(self.df)

    def add_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Next_Day_Return'] = df['Daily Return'].shift(-1)
        df['Target'] = df['Next_Day_Return'].apply(lambda x: 1 if x > 0 else 0)
        return df

    def train_and_evaluate_rf(self):
        base_cols = ['Date', 'Close', 'Daily Return', 'Next_Day_Return']
        cols_to_drop = [col for col in base_cols if col in self.df.columns]
        features = [col for col in self.df.columns if col not in cols_to_drop + ['Target']]
        X = self.df[features]
        y = self.df['Target']

        valid_idx = ~y.isna()
        X, y = X[valid_idx], y[valid_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        model_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_clf.fit(X_train, y_train)

        y_pred = model_clf.predict(X_test)
        print(f"Classification Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        return model_clf

    def train_and_evaluate_rf_regressor(self):
        base_cols = ['Date', 'Daily Return', 'Target']
        cols_to_drop = [col for col in base_cols if col in self.df.columns]
        features = [col for col in self.df.columns if col not in cols_to_drop + ['Next_Day_Return']]
        X = self.df[features]
        y = self.df['Next_Day_Return']

        valid_idx = ~y.isna()
        X, y = X[valid_idx], y[valid_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        model_reg.fit(X_train, y_train)

        y_pred = model_reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Regression MSE: {mse:.4f}")
        print(f"R^2 Score: {r2:.4f}")

        return model_reg

    def predict_tomorrows_trend(self, model):
        base_cols = ['Date', 'Close', 'Daily Return', 'Next_Day_Return']
        cols_to_drop = [col for col in self.df.columns if col in base_cols]
        features = [col for col in self.df.columns if col not in cols_to_drop + ['Target']]

        last_row = self.df.iloc[[-1]]
        prediction = model.predict(last_row[features])[0]
        if prediction == 1:
            return "Price is going up"
        else:
            return "Price is going down"

    def predict_tomorrows_price(self, model):
        base_cols = ['Date', 'Daily Return', 'Target']
        cols_to_drop = [col for col in self.df.columns if col in base_cols]
        features = [col for col in self.df.columns if col not in cols_to_drop + ['Next_Day_Return']]

        last_row = self.df.iloc[[-1]]
        predicted_return = model.predict(last_row[features])[0]
        current_close = last_row['Close'].values[0]
        return current_close * (1 + predicted_return)

# --------------------------------------------------
# Helper Functions for Company Data Processing
# --------------------------------------------------

def process_company_data(company: str, loader: DataLoader) -> pd.DataFrame:
    company_data = AccessCompanyData(company, loader)
    sp = company_data.company_share_price()
    bs = company_data.company_balance_sheet()
    inc = company_data.company_income_statement()
    if sp is not None and bs is not None and inc is not None:
        cleaner = CleanCompanyData(sp, bs, inc)
        enhanced_df = cleaner.get_enhanced_data()
        enhanced_df['Ticker'] = company_data.ticker  # We'll drop it later if needed
        return enhanced_df
    return None

def run_data_pipeline(api_token: str, company_list: list) -> pd.DataFrame:
    loader = DataLoader(api_token=api_token)
    dfs = []
    for comp in company_list:
        df = process_company_data(comp, loader)
        if df is not None:
            dfs.append(df)
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    return pd.DataFrame()

# --------------------------------------------------
# Exposed Functions for the Streamlit App
# --------------------------------------------------

def get_company_data(company: str, api_token: str) -> pd.DataFrame:
    loader = DataLoader(api_token=api_token)
    return process_company_data(company, loader)

def plot_company_data(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Close'], label='Close Price')
    if 'SMA_20' in df.columns:
        ax.plot(df['Date'], df['SMA_20'], label='SMA 20')
    ax.set_title("Company Price Data")
    ax.legend()
    plt.show()
    return fig

def run_ml_model(api_token: str):
    company_list = [
        "American Airlines Group",
        "Delta Air Lines",
        "Southwest Airlines Co.",
        "Spirit Airlines, Inc.",
        "United Airlines Holdings, Inc."
    ]
    merged_df = run_data_pipeline(api_token, company_list)
    if not merged_df.empty:
        normalizer = Normalization(merged_df)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in normalizer.df.columns:
                normalizer.min_max_scale(col)

        scaled_df = normalizer.df
        ml_market = MachineLearning(scaled_df)
        clf_market = ml_market.train_and_evaluate_rf()
        model_reg = ml_market.train_and_evaluate_rf_regressor()
        trend_prediction = ml_market.predict_tomorrows_trend(clf_market)
        price_prediction = ml_market.predict_tomorrows_price(model_reg)
        print("Market Trend Prediction:", trend_prediction)
        print("Market Price Prediction:", price_prediction)
        return price_prediction
    else:
        return None

def run_ml_model_for_company(api_token: str, company: str):
    """
    Fetches the company's historical data, processes it, 
    and runs the random forest regressor to predict tomorrow's price.
    Returns (None, None) if data is insufficient or not found.
    """
    df = get_company_data(company, api_token)
    
    # Check if data is empty or None
    if df is None or df.empty:
        print(f"No data found for {company}. ML prediction cannot be run.")
        return None, None
    
    # If Ticker column is present, drop it
    if 'Ticker' in df.columns:
        df.drop(columns=['Ticker'], inplace=True)

    # Normalize key columns
    normalizer = Normalization(df)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in normalizer.df.columns:
            normalizer.min_max_scale(col)

    # Create a scaled DataFrame and run ML pipeline
    scaled_df = normalizer.df
    ml_market = MachineLearning(scaled_df)
    model_reg = ml_market.train_and_evaluate_rf_regressor()

    # If training or inference fails, you can also catch exceptions
    try:
        price_prediction = ml_market.predict_tomorrows_price(model_reg)
    except Exception as e:
        print(f"Error during ML prediction for {company}: {e}")
        return None, None

    return df, price_prediction


def run_ml_model_for_market(api_token: str):
    company_list = [
        "American Airlines Group",
        "Delta Air Lines",
        "Southwest Airlines Co.",
        "Spirit Airlines, Inc.",
        "United Airlines Holdings, Inc."
    ]
    combined_df = run_data_pipeline(api_token, company_list)
    if not combined_df.empty:
        aggregator = MarketAggregator(combined_df)
        market_df = aggregator.aggregate()

        normalizer = Normalization(market_df)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in normalizer.df.columns:
                normalizer.min_max_scale(col)

        scaled_df = normalizer.df
        ml_market = MachineLearning(scaled_df)
        model_reg = ml_market.train_and_evaluate_rf_regressor()
        price_prediction = ml_market.predict_tomorrows_price(model_reg)
        return market_df, price_prediction
    else:
        return None, None
   

def get_trading_signal(predicted_price: float, current_price: float) -> str:
    """
    Simple trading strategy:
    - BUY if predicted price is >2% higher than current price
    - SELL if predicted price is >2% lower than current price
    - HOLD otherwise
    """
    if predicted_price > current_price * 1.02:
        return "BUY"
    elif predicted_price < current_price * 0.98:
        return "SELL"
    else:
        return "HOLD"


if __name__ == "__main__":
    if api_token:
        print("Running ML pipeline...")
        run_ml_model(api_token)
    else:
        print("Please set your API_KEY in the environment variables.")




