import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
import io
from contextlib import redirect_stdout

# ====== IMPORTS FROM YOUR PREPROCESSING SCRIPT ======
from preprocessingscript_ml import (
    DataLoader,        
    get_company_data,  
    plot_company_data, 
    run_ml_model,
    run_ml_model_for_company,
    run_ml_model_for_market,
    AccessCompanyData,
    get_trading_signal
)

# ------------------------------
# Page configuration & CSS styling
# ------------------------------
st.set_page_config(page_title="Airline Trading Dashboard", layout="wide", page_icon="✈️")

st.markdown("""
<style>
/* ------------------------------
   GLOBAL STYLING
------------------------------ */
body {
    background-color: #f8f9fc;
}
.main .block-container {
    background-color: #ffffff;
    border: 1px solid #e5e5e5;
    padding: 2rem;
    border-radius: 8px;
}
h1, h2, h3, h4, h5, h6 {
    color: #0d3b66;
}
body, p, div, label, span, button {
    color: #343a40;
    font-family: "Open Sans", sans-serif;
}
/* ------------------------------
   SIDEBAR STYLING
------------------------------ */
[data-testid="stSidebar"] {
    background-color: #0d3b66 !important;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
.stRadio label, .stCheckbox label {
    font-weight: 600;
}
/* ------------------------------
   METRIC STYLING
------------------------------ */
div[data-testid="metric-container"] {
    background-color: #f1f3f5; 
    border: 1px solid #dee2e6;
    padding: 15px;
    border-radius: 8px;
    margin: 5px;
}
/* ------------------------------
   TABS STYLING
------------------------------ */
button[data-baseweb="tab"] {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    padding: 10px 20px;
    font-weight: 600;
    color: #0d3b66;
    margin-right: 4px;
}
button[data-baseweb="tab"]:hover {
    background-color: #f5f5f5 !important;
    color: #0d3b66 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #0d3b66 !important;
    color: #ffffff !important;
    border: 1px solid #0d3b66 !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Environment Variables & API Initialization
# ------------------------------
load_dotenv()
api_token = os.getenv('API_KEY')
if not api_token:
    api_token = st.sidebar.text_input("Enter your API key", type="password")

api = None
if api_token:
    api = DataLoader(api_token=api_token)

# ------------------------------
# Caching Functions
# ------------------------------
@st.cache_data(show_spinner=False)
def cached_share_prices(token):
    loader = DataLoader(api_token=token)
    return loader.load_share_prices()

@st.cache_data(show_spinner=False)
def cached_run_ml_for_company(token, company):
    return run_ml_model_for_company(token, company)

@st.cache_data(show_spinner=False)
def cached_run_ml_for_market(token):
    return run_ml_model_for_market(token)

@st.cache_data(show_spinner=False)
def cached_run_ml(token):
    return run_ml_model(token)

# ------------------------------
# Navigation State & Sidebar
# ------------------------------
if "page_index" not in st.session_state:
    st.session_state.page_index = 0

page_options = ["Home", "Go Live", "ML Predictions", "Compare Companies"]
page_index = st.sidebar.radio(
    "Select a Page",
    options=range(len(page_options)),
    index=st.session_state.page_index,
    format_func=lambda i: page_options[i]
)
st.session_state.page_index = page_index
page = page_options[page_index]

# ------------------------------
# HOME PAGE
# ------------------------------
if page == "Home":
    st.title("Welcome to the Airline Trading Dashboard")
    st.image("Main_Logo.jpeg", use_column_width=True)
    st.write("Welcome to the Home Page!")
    st.write("""
    This web application provides real-time and historical stock market data for airline companies,
    along with predictive analytics to help users make informed trading decisions.
    
    *Core Functionalities:*
    - Extract airline stock market data from SimFin.
    - Apply ETL transformations and predictive analytics.
    - Provide trading signals based on model predictions.
    
    *Purpose & Objectives:*
    - Help traders analyze airline stocks.
    - Provide actionable insights using AI-driven predictions.
    """)
    st.subheader("Development Team")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image("Youssef.jpeg", caption="Youssef Abdel Nasser", use_column_width=True)
    with col2:
        st.image("Sergio.jpeg", caption="Sergio Lebed", use_column_width=True)
    with col3:
        st.image("Anastasia.jpeg", caption="Anastasia Chapel", use_column_width=True)
    with col4:
        st.image("Felix.jpeg", caption="Felix Goosens", use_column_width=True)
    with col5:
        st.image("Marta.jpeg", caption="Marta Perez", use_column_width=True)
    st.write("### Airline Logos")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        if st.button("American Airlines", key="btn_aal"):
            st.session_state.selected_ticker = "AAL"
            st.session_state.page_index = 1
            st.experimental_rerun()
        st.image("American_Airlines_logo.jpeg", use_column_width=True)
    with col_b:
        if st.button("Spirit Airlines", key="btn_spirit"):
            st.session_state.selected_ticker = "SAVE"
            st.session_state.page_index = 1
            st.experimental_rerun()
        st.image("Spirit_logo.jpeg", use_column_width=True)
    with col_c:
        if st.button("United Airlines", key="btn_ual"):
            st.session_state.selected_ticker = "UAL"
            st.session_state.page_index = 1
            st.experimental_rerun()
        st.image("United_logo.jpeg", use_column_width=True)
    with col_d:
        if st.button("Delta Airlines", key="btn_dal"):
            st.session_state.selected_ticker = "DAL"
            st.session_state.page_index = 1
            st.experimental_rerun()
        st.image("Delta_logo.jpeg", use_column_width=True)
    with col_e:
        if st.button("Southwest Airlines", key="btn_luv"):
            st.session_state.selected_ticker = "LUV"
            st.session_state.page_index = 1
            st.experimental_rerun()
        st.image("SouthWest_Logo.jpeg", use_column_width=True)

# ------------------------------
# GO LIVE PAGE
# ------------------------------
elif page == "Go Live":
    st.title("📈 Real-Time Market Analysis")
    
    # --- Ticker Lookup via Company Name ---
    company_input = st.text_input("Enter Company Name (optional)", value="")
    if company_input:
        try:
            if api_token:
                loader = DataLoader(api_token=api_token)
                acd = AccessCompanyData(company_input, loader)
                found_ticker = acd.ticker
                if found_ticker:
                    st.info(f"Ticker for '{company_input}' found: {found_ticker}")
                    default_ticker = found_ticker
                else:
                    st.warning(f"No ticker found for '{company_input}'. Using default ticker.")
                    default_ticker = st.session_state.get("selected_ticker", "AAL")
            else:
                default_ticker = st.session_state.get("selected_ticker", "AAL")
        except Exception as e:
            st.error(f"Error finding ticker: {e}")
            default_ticker = st.session_state.get("selected_ticker", "AAL")
    else:
        default_ticker = st.session_state.get("selected_ticker", "AAL")
    
    ticker_list = ["AAL", "DAL", "UAL", "LUV", "SAVE"]
    try:
        default_index = ticker_list.index(default_ticker)
    except ValueError:
        default_index = 0

    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.selectbox("Select Airline Stock", ticker_list, index=default_index, label_visibility="collapsed")
        
        # Key period radio
        time_period = st.radio(
            "Time Period",
            ["5D", "10D", "1M", "YTD", "Custom"],
            horizontal=True,
            help="Select a predefined period or choose 'Custom' to pick your own dates"
        )
        
        # Also have a slider to override the number of days if desired
        override_days = st.slider("Override Days (optional)", min_value=1, max_value=365, value=30, step=1,
                                  help="Use this slider to override the selected time period with a custom number of days")
        
        share_prices = cached_share_prices(api_token)
        share_prices = share_prices[share_prices["Ticker"] == ticker]
        if not share_prices.empty:
            final_date = pd.to_datetime(share_prices["Date"]).max()
        else:
            final_date = datetime.today()

        # Default start_date logic based on radio
        if time_period == "5D":
            start_date = final_date - timedelta(days=5)
        elif time_period == "10D":
            start_date = final_date - timedelta(days=10)
        elif time_period == "1M":
            start_date = final_date - relativedelta(months=1)
        elif time_period == "YTD":
            start_date = datetime(final_date.year, 1, 1)
        else:
            # Custom
            start_date = st.date_input("Start Date", datetime(2020, 1, 1))
            end_date_custom = st.date_input("End Date", final_date)
        
        # If user moved the slider, override the start_date based on it
        # (unless user is in 'Custom' mode)
        if time_period != "Custom":
            start_date = final_date - timedelta(days=override_days)
            end_date = final_date
        else:
            end_date = end_date_custom
        
        st.caption(f"Date Range: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
    with col2:
        st.markdown("### Info")
        st.info("Use the slider to override any chosen period. If 'Custom' is selected, the slider won't apply.")

    if st.button("🚀 Fetch Market Data", type="primary"):
        if api_token:
            data = cached_share_prices(api_token)
            data = data[data["Ticker"] == ticker]
            data = data[(data["Date"] >= pd.to_datetime(start_date)) & 
                        (data["Date"] <= pd.to_datetime(end_date))]
            data = data.sort_values("Date")
            if not data.empty:
                current_close = data['Close'].iloc[-1]
                open_price = data['Open'].iloc[-1]
                price_change = current_close - open_price
                pct_change = (price_change / open_price) * 100
                day_low = data['Low'].iloc[-1]
                day_high = data['High'].iloc[-1]
                day_range = f"{day_low:.2f} - {day_high:.2f}"
                volume = data['Volume'].iloc[-1]

                metric_cols = st.columns(3)
                metric_cols[0].metric("Current Price", f"${current_close:.2f}",
                                      f"{price_change:.2f} ({pct_change:.2f}%)")
                metric_cols[1].metric("Day Range", day_range)
                metric_cols[2].metric("Volume", f"{volume:,}")

                # Original interactive chart
                tab1, tab2 = st.tabs(["Interactive Chart", "Historical Data"])
                with tab1:
                    fig = px.line(data, x="Date", y="Close", title=f"{ticker} Price Chart")
                    fig.update_layout(
                        xaxis=dict(rangeslider=dict(visible=True), tickformat="%d/%m/%Y"),
                        yaxis_title="Price (USD)",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Use the modebar on the chart to download as image.")
                with tab2:
                    styled_data = data.style.format({
                        'Open': '${:.2f}',
                        'High': '${:.2f}',
                        'Low': '${:.2f}',
                        'Close': '${:.2f}',
                        'Volume': '{:,}'
                    })
                    st.dataframe(styled_data, use_container_width=True)
                
                # Download button to export data as CSV
                csv_data = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Data as CSV",
                    data=csv_data,
                    file_name=f"{ticker}_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No market data available for the selected parameters.")
        else:
            st.error("Please provide a valid API key in the sidebar.")

# ------------------------------
# ML PREDICTIONS PAGE
# ------------------------------
elif page == "ML Predictions":
    st.title("🤖 ML Predictions Overview")
    st.write("This section shows ML-based predictions along with interactive visualizations.")

    # Mapping from ticker to full company names, plus "Market Total"
    company_mapping = {
        "AAL": "American Airlines Group",
        "DAL": "Delta Air Lines",
        "UAL": "United Airlines Holdings, Inc.",
        "LUV": "Southwest Airlines Co.",
        "SAVE": "Spirit Airlines, Inc.",
        "Market Total": "Market Total"
    }

    ticker_list = list(company_mapping.keys())
    default_ticker = st.session_state.get("selected_ticker", "AAL")
    if default_ticker not in ticker_list:
        default_ticker = "AAL"
    try:
        default_index = ticker_list.index(default_ticker)
    except ValueError:
        default_index = 0

    ticker = st.selectbox("Select Company or Market", ticker_list, index=default_index)
    full_company_name = company_mapping[ticker]

    # Run the ML predictions when button is clicked
    if st.button("Run ML Predictions"):
        if not api_token:
            st.error("Please provide a valid API key in the sidebar.")
        else:
            with st.spinner(f"Running ML Pipeline for {full_company_name}..."):
                if ticker == "Market Total":
                    historic_data, ml_price_prediction = cached_run_ml_for_market(api_token)
                else:
                    historic_data, ml_price_prediction = cached_run_ml_for_company(api_token, full_company_name)
            
            # Check for None or empty data
            if historic_data is None or historic_data.empty or ml_price_prediction is None:
                st.warning("ML prediction could not be generated for this company. It may not have sufficient historical data.")
            else:
                st.success(f"Predicted Price for Tomorrow ({full_company_name}): ${ml_price_prediction:.2f}")
                # Trading signal logic
                current_price = historic_data['Close'].iloc[-1]
                trading_signal = get_trading_signal(ml_price_prediction, current_price)
                st.info(f"Trading Signal: **{trading_signal}** ")

                st.session_state.ml_prediction = ml_price_prediction
                st.session_state.ml_historic_data = historic_data
                st.session_state.ml_company_name = full_company_name

    # Display prediction chart only if valid prediction exists
    if ("ml_prediction" in st.session_state and "ml_historic_data" in st.session_state 
        and st.session_state.ml_historic_data is not None and not st.session_state.ml_historic_data.empty):
        data = st.session_state.ml_historic_data.sort_values("Date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],
                                 mode='lines', name='Historical Price'))
        tomorrow = data['Date'].max() + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(x=[tomorrow], y=[st.session_state.ml_prediction],
                                 mode='markers', marker=dict(size=12),
                                 name='Predicted Price'))
        fig.update_layout(
            title=f"{st.session_state.ml_company_name} Price History with ML Prediction",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional chart: Plot RSI if available
        if "RSI" in data.columns:
            st.subheader("RSI Over Time")
            fig_rsi = px.line(data, x="Date", y="RSI", title="Relative Strength Index (RSI)")
            fig_rsi.update_layout(
                xaxis=dict(rangeslider=dict(visible=True), tickformat="%d/%m/%Y"),
                yaxis_title="RSI",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

# ------------------------------
# COMPARE COMPANIES PAGE
# ------------------------------
elif page == "Compare Companies":
    st.title("📊 Compare Companies")
    st.write("Select two or more companies to compare their closing prices side-by-side over a specified date range.")

    # Multi-select for companies to compare
    compare_options = ["AAL", "DAL", "UAL", "LUV", "SAVE"]
    selected_companies = st.multiselect("Select companies", compare_options, default=["AAL", "DAL"])
    
    # Date selection logic
    st.subheader("Select Date Range for Comparison")
    time_period = st.radio(
        "Time Period",
        ["5D", "10D", "1M", "YTD", "Custom"],
        horizontal=True,
        help="Select a predefined period or choose 'Custom' to pick your own dates"
    )
    override_days = st.slider("Override Days (optional)", min_value=1, max_value=365, value=30, step=1,
                              help="Use this slider to override the selected time period with a custom number of days")

    compare_button = st.button("Compare Data", type="primary")

    if compare_button:
        if not selected_companies:
            st.warning("Please select at least one company to compare.")
        elif not api_token:
            st.warning("No valid API key provided. Data cannot be fetched.")
        else:
            # Fetch data
            share_prices = cached_share_prices(api_token)
            if share_prices.empty:
                st.warning("No data returned from the API.")
            else:
                # Filter to selected companies
                combined_data = pd.DataFrame()
                for comp in selected_companies:
                    comp_data = share_prices[share_prices["Ticker"] == comp]
                    if not comp_data.empty:
                        comp_data = comp_data.copy()
                        comp_data["Date"] = pd.to_datetime(comp_data["Date"])
                        comp_data["Ticker"] = comp
                        combined_data = pd.concat([combined_data, comp_data])

                combined_data.sort_values("Date", inplace=True)

                if not combined_data.empty:
                    final_date = combined_data["Date"].max()
                else:
                    final_date = datetime.today()

                # Determine start/end date based on radio
                if time_period == "5D":
                    start_date = final_date - timedelta(days=5)
                    end_date = final_date
                elif time_period == "10D":
                    start_date = final_date - timedelta(days=10)
                    end_date = final_date
                elif time_period == "1M":
                    start_date = final_date - relativedelta(months=1)
                    end_date = final_date
                elif time_period == "YTD":
                    start_date = datetime(final_date.year, 1, 1)
                    end_date = final_date
                else:
                    # Custom
                    custom_col1, custom_col2 = st.columns(2)
                    with custom_col1:
                        start_date = st.date_input("Start Date", datetime(2020, 1, 1))
                    with custom_col2:
                        end_date = st.date_input("End Date", final_date)

                # If not custom, allow slider override
                if time_period != "Custom":
                    start_date = final_date - timedelta(days=override_days)

                st.caption(f"Comparison Date Range: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")

                # Filter combined_data by date range
                mask = (combined_data["Date"] >= pd.to_datetime(start_date)) & \
                       (combined_data["Date"] <= pd.to_datetime(end_date))
                combined_data = combined_data[mask]

                if not combined_data.empty:
                    fig_compare = px.line(
                        combined_data,
                        x="Date",
                        y="Close",
                        color="Ticker",
                        title="Comparison of Closing Prices"
                    )
                    fig_compare.update_layout(
                        xaxis=dict(rangeslider=dict(visible=True), tickformat="%d/%m/%Y"),
                        yaxis_title="Price (USD)",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)

                    csv_compare = combined_data.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Comparison Data as CSV",
                        data=csv_compare,
                        file_name="comparison_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data available for the selected companies in this date range.")
    else:
        st.info("Click **Compare Data** to generate the chart and data download for your selected companies.")
