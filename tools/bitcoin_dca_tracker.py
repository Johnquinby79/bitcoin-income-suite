import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf  # For live/historical BTC and S&P data
import sys
sys.path.append('.')
from utils.dgi import calculate_dgi, display_dgi  # Shared DGI util
from utils.bitcoin_utils import calculate_cagr  # Shared for CAGR efficiencies

# Fetch Historical BTC for Realistic Growth (CAGR)
try:
    btc_hist = yf.download('BTC-USD', period='5y')['Close']
    hist_cagr = calculate_cagr(btc_hist)  # Use shared util
    btc_growth_default = round(hist_cagr, 1) if not np.isnan(hist_cagr) else 50.0
except:
    btc_growth_default = 50.0

# Fetch Historical S&P for Comparison (CAGR ~10%)
try:
    sp_hist = yf.download('^GSPC', period='5y')['Close']
    sp_growth_default = round(calculate_cagr(sp_hist), 1) if not np.isnan(calculate_cagr(sp_hist)) else 10.0
except:
    sp_growth_default = 10.0

savings_rate = 4.0  # Annual % for savings account

# App Title
st.title("Bitcoin DCA Simulator for Beginners")

st.markdown("""
DCA stands for Dollar Cost Averaging, a simple way to save by buying a fixed amount of Bitcoin regularly, no matter the price. It builds the habit of consistent saving instead of spending, helping you ignore short-term ups and downs. Over time, this can grow your money steadily through Bitcoin's potential, buying back your future time by giving you financial freedom to work less and live more.
""")

# Onboarding Questionnaire (Full for Assessment/Tiering)
with st.form("Onboarding Questionnaire"):
    st.subheader("Quick Assessment: Tailor Your DCA Plan")
    literacy = st.selectbox("Financial Literacy Level", ["Beginner", "Intermediate", "Advanced"])
    goal = st.selectbox("Main Goal", ["Learn Basics: Start understanding how Bitcoin works as a way to save and grow money.", "Build Wealth: Focus on long-term growth by regularly adding to your Bitcoin savings.", "Generate Income: Use Bitcoin and related investments to create ongoing earnings."])
    budget = st.number_input("Disposable Monthly Income ($)", 100, 2000, 500, step=50)
    # Add more for 8-10 questions (e.g., risk, horizon prefs)
    risk_level = st.selectbox("Risk Tolerance", ["Low: Prefer safer options such as saving small amounts in Bitcoin over time.", "Medium: Will save larger amounts in Bitcoin and am willing to learn other techniques.", "High: I am ready, understanding and managing risk is part of the game."])
    horizon_pref = st.number_input("Preferred Horizon (Years)", 1, 30, 5)
    submit = st.form_submit_button("Generate Personalized Plan")

if submit:
    dca_percent_suggest = 5 if literacy == "Beginner" else 5 if literacy == "Intermediate" else 10
    suggested_dca = round(budget * (dca_percent_suggest / 100) / 5) * 5  # Round to nearest $5
    asset_rec = "Bitcoin"  # Default to direct for simplicity
    plan_text = f"Save {dca_percent_suggest}% of your spare money (${suggested_dca} each month) in {asset_rec} and start to {goal.split(':')[0].strip()}."
    st.success(f"Your Plan: {plan_text} Plan for {horizon_pref} years. Your Starting DGI Score: 50 – DGI stands for Delayed Gratification Index, which tracks how well you're choosing long-term saving over spending now. Check back to see it improve!")
    if literacy == "Advanced":
        st.info("Advanced Tip: Layer with leverage from Covered Call Recommender for premium-funded DCA.")
    st.session_state['suggested_dca'] = max(5.0, suggested_dca)  # Clamp to min 5.0
    st.session_state['suggested_horizon'] = horizon_pref
    st.session_state['dca_percent_suggest'] = dca_percent_suggest
    st.session_state['literacy'] = literacy

# Inputs (Preset from Questionnaire, Tiered by Literacy)
preset_dca = float(st.session_state.get('suggested_dca', 100.0))  # Cast to float
if st.session_state.get('literacy', "Beginner") == "Beginner":
    monthly_dca = st.slider("Monthly DCA Amount ($)", min_value=5.0, max_value=1000.0, value=preset_dca, step=5.0, help="Amount to put into Bitcoin each month (in $5 steps for simplicity).")
else:
    monthly_dca = st.number_input("Monthly DCA Amount ($)", min_value=5.0, max_value=1000.0, value=preset_dca, step=1.0, help="Amount to put into Bitcoin each month (fine-tune as needed).")
if monthly_dca < 20:
    st.info("Starting small is great for building habits—consider increasing to $20+ to speed up Bitcoin's growth compared to regular savings!")

time_horizon = st.number_input("How many years do you want to plan for? (e.g., how long do you plan on saving in Bitcoin)", min_value=1, max_value=30, value=st.session_state.get('suggested_horizon', 10), step=1)
btc_growth_rate = st.slider("Bitcoin Growth Rate (%)", min_value=10, max_value=80, value=int(btc_growth_default), step=1, help=f"Historical average ~{btc_growth_default}%; adjust for conservatism.") / 100
include_diminishing = st.checkbox("Make growth slower over time? (e.g., Assets grow slower as they get bigger over time.)", value=True)
backtest = st.checkbox("Backtest with Historical Data?", value=False, help="Use real BTC data for past performance simulation.")

# Simulation Function (with Backtest Logic and Comparisons)
def simulate_dca(monthly_dca, horizon, growth_rate, diminishing, backtest):
    months = horizon * 12
    month_list = list(range(months + 1))
    dca_list = [0] + [monthly_dca] * months
    btc_value = [0]
    sp_value = [0]
    savings_value = [0]
    if backtest:
        if len(btc_hist) < 2 or len(sp_hist) < 2:
            st.warning("Insufficient historical data for backtest—using forward simulation.")
            backtest = False
        else:
            # Redesign: Convert to list of floats for scalar access
            monthly_btc_list = btc_hist.resample('M').last().fillna(method='ffill').astype(float).squeeze().to_list()[-min(months, len(btc_hist)):]
            monthly_sp_list = sp_hist.resample('M').last().fillna(method='ffill').astype(float).squeeze().to_list()[-min(months, len(sp_hist)):]
            avg_btc_growth = growth_rate / 12
            avg_sp_growth = sp_growth_default / 1200  # Monthly
            savings_monthly_rate = savings_rate / 1200
            prev_btc = 0
            prev_sp = 0
            prev_savings = 0
            for m in range(1, months + 1):
                if m < len(monthly_btc_list):
                    prev_price_btc = monthly_btc_list[m-1]
                    curr_price_btc = monthly_btc_list[m]
                    btc_growth = (curr_price_btc / prev_price_btc - 1) if prev_price_btc != 0 else avg_btc_growth
                else:
                    btc_growth = avg_btc_growth
                new_btc = (prev_btc + monthly_dca) * (1 + btc_growth)
                btc_value.append(new_btc)
                prev_btc = new_btc

                if m < len(monthly_sp_list):
                    prev_price_sp = monthly_sp_list[m-1]
                    curr_price_sp = monthly_sp_list[m]
                    sp_growth = (curr_price_sp / prev_price_sp - 1) if prev_price_sp != 0 else avg_sp_growth
                else:
                    sp_growth = avg_sp_growth
                new_sp = (prev_sp + monthly_dca) * (1 + sp_growth)
                sp_value.append(new_sp)
                prev_sp = new_sp

                new_savings = (prev_savings + monthly_dca) * (1 + savings_monthly_rate)
                savings_value.append(new_savings)
                prev_savings = new_savings
    else:
        avg_btc_growth = growth_rate / 12
        avg_sp_growth = sp_growth_default / 1200
        savings_monthly_rate = savings_rate / 1200
        prev_btc = 0
        prev_sp = 0
        prev_savings = 0
        for m in range(1, months + 1):
            effective_btc = avg_btc_growth * (0.95 ** (m / 12)) if diminishing else avg_btc_growth
            new_btc = (prev_btc + monthly_dca) * (1 + effective_btc)
            btc_value.append(new_btc)
            prev_btc = new_btc

            new_sp = (prev_sp + monthly_dca) * (1 + avg_sp_growth)
            sp_value.append(new_sp)
            prev_sp = new_sp

            new_savings = (prev_savings + monthly_dca) * (1 + savings_monthly_rate)
            savings_value.append(new_savings)
            prev_savings = new_savings

    df = pd.DataFrame({
        'Month': month_list,
        'Monthly DCA ($)': dca_list,
        'BTC Value ($)': btc_value,
        'S&P 500 Value ($)': sp_value,
        'Savings Account Value ($)': savings_value
    })
    return df

# Run Simulation
if st.button("Run DCA Simulation"):
    df = simulate_dca(monthly_dca, time_horizon, btc_growth_rate, include_diminishing, backtest)
    st.subheader("DCA Projection Table")
    st.dataframe(df.style.format("{:.2f}").hide(axis="index"))  # Hide index, format USD

    # Chart (Mobile-Friendly Size)
    fig, ax = plt.subplots(figsize=(6, 4))  # Optimized for mobile
    ax.plot(df['Month'], df['BTC Value ($)'], label='BTC Value')
    ax.plot(df['Month'], df['S&P 500 Value ($)'], label='S&P 500 Value')
    ax.plot(df['Month'], df['Savings Account Value ($)'], label='Savings Account Value')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value ($)')
    ax.set_title('Bitcoin DCA Forecast vs. Alternatives')
    ax.legend()
    st.pyplot(fig)

    # Enhanced DGI (Action Points Formula + Badge)
    if 'dgi_score' not in st.session_state:
        st.session_state.dgi_score = 50
    action_points = 10 + (st.session_state.get('dca_percent_suggest', 5) - 2) * 2  # + for higher suggest %
    st.session_state.dgi_score = calculate_dgi(action_points, st.session_state.dgi_score)
    display_dgi()
    if st.session_state.dgi_score > 70:
        st.success("Badge Earned: DCA Pro! Consistent Bitcoin saving unlocked.")
    if 'dca_runs' not in st.session_state:
        st.session_state['dca_runs'] = 0
    st.session_state['dca_runs'] += 1
    if st.session_state['dca_runs'] > 3:
        st.success("Badge Earned: Streak Master! 3+ simulations—great discipline.")

    # Export for Suite (with Enhanced Context)
    final_value = df['BTC Value ($)'].iloc[-1]
    total_invested = monthly_dca * time_horizon * 12
    annualized_yield = ((final_value / total_invested) ** (1 / time_horizon) - 1) * 100 if time_horizon > 0 else 0
    export_text = f"{annualized_yield:.2f}"
    st.text_area("Copy this number (your expected yearly growth rate from the simulation) and paste it into other tools like the Income Sweep Simulator. This lets you see how combining your Bitcoin savings with earnings from stocks or options can make your money grow even faster.", export_text, height=50)

    # Suite Integration Expander
    with st.expander("Suite Integration"):
        st.markdown("""
        Export yield to [Income Sweep Simulator](https://github.com/Johnquinby79/bitcoin-income-suite/blob/main/tools/income_sweep_simulator.py) for dividend layering with leverage; 
        [Covered Call Recommender](https://github.com/Johnquinby79/bitcoin-income-suite/blob/main/tools/Covered_Call_Options_Recommender.py) for premiums funding DCA.
        """)

    # PNSD Alignment Expander
    with st.expander("North Star Reference"):
        st.markdown("This tool aligns with the Project North Star Document (PNSD) [link to Google Doc]. It supports goals like educating on Bitcoin's compounding and integrating with recommenders for premium-funded flywheels, with DGI for behavioral tracking.")

    # Disclaimer Footer
    st.markdown("**Disclaimer:** Educational only; volatility/taxes apply—consult pros. Focus on discipline for Bitcoin's long-term power.")

# Last Update
st.write(f"Last data refresh: {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}")