import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('.')  # Ensure utils folder is in path for imports
try:
    from utils.bitcoin_utils import project_btc_price  # Shared projection function
except ModuleNotFoundError:
    st.error("Utils module not found. Ensure utils/bitcoin_utils.py exists with the project_btc_price function.")
    # Fallback inline function if utils fails (temporary for testing)
    def project_btc_price(start_price=118000.0, start_year=2025, target_year=2045, historical_cagr=0.8220, decay_factor=0.8525):
        years_ahead = target_year - start_year + 1
        prices = {start_year: start_price}
        P = start_price
        for t in range(1, years_ahead):
            rt = historical_cagr * (decay_factor ** (t - 1))
            P = P * (1 + rt)
            prices[start_year + t] = round(P, 2)
        return prices

from utils.dgi import calculate_dgi, display_dgi  # Shared DGI util

# App Title
st.title("Bitcoin Opportunity Cost Calculator")

st.markdown("""
This tool explains why spending money today can cost you a lot in time and money later. When you buy something enjoyable now, you miss the chance to grow that money in Bitcoin. Bitcoin acts like a smart savings tool that can increase your money over time. Check what your spent money might turn into in dollars or how many work hours it could save you later. It guides you to make wise saving choices for greater wealth and future freedom!
""")

# Onboarding Questionnaire (Mobile-Friendly Columns)
col1, col2 = st.columns(2)
with col1:
    literacy = st.selectbox("Financial Literacy Level", ["Beginner", "Intermediate", "Advanced"], help="Your experience level: Beginner (new to money stuff, like just learning to save), Intermediate (know basics like saving vs. spending), Advanced (understand investing and Bitcoin growth). This helps suggest smart choices.")
    hourly_wage = st.number_input("Hourly Wage ($)", 10, 100, 25, step=5)
with col2:
    spend_amount = st.number_input("Amount to Spend/Save ($)", 10, 10000, 100)
    time_horizon = st.slider("Time Horizon (Years)", 1, 30, 10, help="How many years into the future? Longer time shows Bitcoin's magic in growing money, but remember, growth slows a bit over time.")

# Simulation Function (Always Using Diminishing Returns Model)
def calculate_opp_cost(amount, horizon, wage, start_year=2025):
    years = np.arange(0, horizon + 1)
    df = pd.DataFrame({'Year': start_year + years, 'Future Value in Bitcoin ($)': np.zeros(len(years))})
    df.loc[0, 'Future Value in Bitcoin ($)'] = amount
    prices = project_btc_price()  # Get shared model prices (always diminishing)
    for y in range(1, horizon + 1):
        year = start_year + y
        if year in prices and (year - 1) in prices:
            rt = (prices[year] / prices[year - 1]) - 1  # Derive annual rt from model
        else:
            rt = 0.03  # Flat conservative if beyond model (2045+)
        prev_value = df.loc[y-1, 'Future Value in Bitcoin ($)']
        new_value = prev_value * (1 + rt)
        df.loc[y, 'Future Value in Bitcoin ($)'] = new_value
    df['Future Work Hours Saved'] = df['Future Value in Bitcoin ($)'] / wage
    df['Year'] = df['Year'].astype(int)
    return df

if st.button("Run Calculation"):
    df = calculate_opp_cost(spend_amount, time_horizon, hourly_wage)
    st.subheader("Projection Table")
    st.dataframe(df.round(2))

    # Chart (Responsive for Mobile, BTC-Only)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['Year'], df['Future Value in Bitcoin ($)'], label='Bitcoin Value')
    ax.set_xlabel('Year')
    ax.set_ylabel('Future Value ($)')
    ax.set_title('What Your Money Could Grow To in Bitcoin')
    ax.legend()
    st.pyplot(fig)

    # Enhanced DGI (+points based on literacy/recommendation for delayed choice)
    if 'dgi_score' not in st.session_state:
        st.session_state.dgi_score = 50
    action_points = 10 if literacy != "Beginner" else 5  # Higher for advanced delayed mindset
    st.session_state.dgi_score = calculate_dgi(action_points, st.session_state.dgi_score)
    display_dgi()
    if st.session_state.dgi_score > 70:
        st.success("Badge Earned: Wise Saver! Shifted toward Bitcoin delayed gratification.")

    # Export for Suite Efficiency
    hours_saved = df['Future Work Hours Saved'].iloc[-1]
    export_text = f"{hours_saved:.2f}"
    st.text_area("Copy Future Work Hours Saved for DCA Tracker/Simulator", export_text, height=50, help="Paste into DCA for consistent savings sims funded by avoided spending, or Simulator for sweeps with leverage at 20-50% allocations.")

    # Suite Integration Expander
    with st.expander("Suite Integration"):
        st.markdown("""
        Export to [DCA Tracker](https://github.com/Johnquinby79/bitcoin-income-suite/blob/main/tools/bitcoin_dca_tracker.py) for payroll budgeting; 
        [Income Sweep Simulator](https://github.com/Johnquinby79/bitcoin-income-suite/blob/main/tools/income_sweep_simulator.py) for dividend/premium layering into BTC.
        """)

    # PNSD Alignment Expander
    with st.expander("North Star Reference"):
        st.markdown("Aligns with PNSD [link to Google Doc]: Educates on BTC vs. spending for behavioral change, with DGI tracking shifts to delayed gratification.")

# Assumptions Expander (Updated to Reflect Always-On Diminishing)
with st.expander("Assumptions: How This Tool Works"):
    st.markdown("""
    - Bitcoin growth starts at ~82% a year (from the last 10 years), slowing to hit $10 million by 2045—like a fast start that eases up.
    - Growth slows down over years (diminishing returns): Like running fast at first, then tiring a bit. We reduce growth to be safe and real.
    - Future Value: If you save $100 in Bitcoin instead of spending, we show how much it could be worth in dollars after your chosen years.
    - Future Work Hours Saved: Divides the grown money by your hourly wage. Example: If it grows to $500 and you earn $25/hour, you save 20 hours of work later!
    - Ties to Bitcoin power: Saving in Bitcoin helps it grow your money like an income machine, better than keeping cash that loses value over time.
    """)

# Disclaimer & Last Update
st.markdown("**Disclaimer:** Educational only; volatility/taxes apply—consult pros. Focus on Bitcoin's long-term income potential.")
st.write(f"Last data refresh: {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}")