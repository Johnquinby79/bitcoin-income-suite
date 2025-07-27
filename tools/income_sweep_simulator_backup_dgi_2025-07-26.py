import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import sys
sys.path.append('.')
from utils.dgi import calculate_dgi, display_dgi  # Shared DGI util

def simulate_income_sweep(portfolio_value=10000, dividend_yield=0.03, sweep_percentage=0.5,
                          time_horizon=10, stock_growth_rate=0.07, btc_growth_rate=0.15,
                          options_premium_yield=0.0, premium_sweep_percentage=0.5,
                          strategy_type='None'):
    """
    Simulates Bitcoin income sweeps from dividends (basic) and options premiums (advanced).
    Returns a DataFrame with yearly projections and a matplotlib figure.
    Emphasizes Bitcoin as income generator via disciplined sweeps.
    """
    years = np.arange(0, time_horizon + 1)
    data = {'Year': years, 'Portfolio Value': np.zeros(len(years)),
            'Annual Dividends': np.zeros(len(years)), 'Swept to BTC (Div)': np.zeros(len(years)),
            'Annual Premiums': np.zeros(len(years)), 'Swept to BTC (Prem)': np.zeros(len(years)),
            'BTC Holdings Value': np.zeros(len(years)), 'Total Wealth': np.zeros(len(years))}
    df = pd.DataFrame(data)

    df.loc[0, 'Portfolio Value'] = portfolio_value
    df.loc[0, 'Total Wealth'] = portfolio_value

    for year in range(1, time_horizon + 1):
        prev_portfolio = df.loc[year-1, 'Portfolio Value']
        dividends = prev_portfolio * dividend_yield
        swept_div = dividends * sweep_percentage
        premiums = prev_portfolio * options_premium_yield if options_premium_yield > 0 else 0
        swept_prem = premiums * premium_sweep_percentage
        total_swept = swept_div + swept_prem

        # Reinvest non-swept into portfolio
        non_swept = (dividends - swept_div) + (premiums - swept_prem)
        new_portfolio = (prev_portfolio + non_swept) * (1 + stock_growth_rate)

        # BTC growth (simplified: assumes sweeps buy BTC at current value, grows uniformly)
        prev_btc = df.loc[year-1, 'BTC Holdings Value']
        new_btc = (prev_btc + total_swept) * (1 + btc_growth_rate)

        df.loc[year, 'Portfolio Value'] = new_portfolio
        df.loc[year, 'Annual Dividends'] = dividends
        df.loc[year, 'Swept to BTC (Div)'] = swept_div
        df.loc[year, 'Annual Premiums'] = premiums
        df.loc[year, 'Swept to BTC (Prem)'] = swept_prem
        df.loc[year, 'BTC Holdings Value'] = new_btc
        df.loc[year, 'Total Wealth'] = new_portfolio + new_btc

    # Create figure for visualization
    fig, ax = plt.subplots()
    ax.plot(df['Year'], df['Total Wealth'], label='Total Wealth')
    ax.plot(df['Year'], df['Portfolio Value'], label='Portfolio (Stocks/ETFs)')
    ax.plot(df['Year'], df['BTC Holdings Value'], label='Bitcoin Holdings')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value ($)')
    ax.set_title(f'Bitcoin Income Sweep Simulation\nStrategy: {strategy_type}')
    ax.legend()
    plt.close(fig)  # Prevents duplicate displays in Streamlit

    return df, fig

# Streamlit UI
st.title("Bitcoin Income Sweep Simulator for Beginners")
st.markdown("""
This tool visualizes channeling dividend income from ETFs/stocks into Bitcoin for compounding growth, highlighting Bitcoin's potential as an income generator. Use basic mode for simple dividend sweeps, or expand for advanced options premiums—integrating efficiencies from your Covered Call Options Recommender or Cash Secured Put Recommender. Emphasize discipline: e.g., 20-50% allocations to sweeps, avoiding over-leverage, and focusing on time-based wealth creation.
""")

# Basic inputs
portfolio_value = st.slider("Starting Portfolio Value ($)", 1000, 100000, 10000)
dividend_yield = st.slider("Annual Dividend Yield (%)", 1.0, 5.0, 3.0) / 100
sweep_percentage = st.slider("Dividend Sweep Percentage to Bitcoin (%)", 0, 100, 50) / 100
time_horizon = st.slider("Time Horizon (Years)", 1, 30, 10)
stock_growth_rate = st.slider("Stock/ETF Growth Rate (%)", 5.0, 10.0, 7.0) / 100
btc_growth_rate = st.slider("Bitcoin Growth Rate (%)", 10.0, 30.0, 15.0) / 100  # Conservative historical proxy

# Advanced expander
with st.expander("Advanced Users: Integrate Options Premiums"):
    st.markdown("""
    For experienced investors, incorporate options premiums from strategies like covered calls or cash-secured puts (leverage your Covered Call Options Recommender or Cash Secured Put Recommender for tailored trade suggestions). Sweep these into Bitcoin to supercharge the flywheel—e.g., sell puts on ETFs like SPY for 1-5% monthly premiums, auto-convert to BTC, and maintain disciplined allocations. This builds on dividends for steady, leveraged income generation without overextending.
    
    **Pros/Cons Table:**
    
    | Aspect | Pros | Cons | Mitigation |
    |--------|------|------|------------|
    | Yield Boost | Combines 5-15% premiums from puts/calls with dividends for higher sweeps. | Risk of assignment if the underlying drops. | Use your Cash Secured Put Recommender for conservative strikes; allocate only what you can afford to buy. |
    | Bitcoin Sweep | Premiums accelerate BTC accumulation in the suite's flywheel. | Tax on premiums as income. | Employ tax-efficient accounts; sweep via tools like your recommenders' integrations. |
    | Discipline | Promotes regular, rule-based trading. | Market volatility impacting premiums. | Set min_yield filters in recommenders; review quarterly for allocations. |
    | Suite Efficiency | Import yields from Covered Call or Cash Secured Put Recommenders for accurate sims. | Requires tool familiarity. | Start with presets, then customize with recommender data for efficient wealth building. |
    
    **Efficiency Tip:** Pull premium projections from your Cash Secured Put Recommender (e.g., 12% annualized on QQQ) and input here—simulate sweeping 60% into Bitcoin while using 40% to roll positions, tying into intelligent leverage for sustained income.
    """)
    options_premium_yield = st.slider("Options Premium Yield (%)", 0.0, 20.0, 0.0) / 100
    premium_sweep_percentage = st.slider("Premium Sweep to Bitcoin (%)", 0, 100, 50) / 100
    strategy_type = st.selectbox("Strategy Type", ["None", "Covered Calls", "Cash Secured Puts"])

# Run and display simulation
df, fig = simulate_income_sweep(portfolio_value, dividend_yield, sweep_percentage, time_horizon,
                                stock_growth_rate, btc_growth_rate, options_premium_yield,
                                premium_sweep_percentage, strategy_type)

st.subheader("Projection Table")
st.dataframe(df.round(2))

st.subheader("Growth Chart")
st.pyplot(fig)

cagr = ((df['Total Wealth'].iloc[-1] / df['Total Wealth'].iloc[0]) ** (1 / time_horizon) - 1) * 100 if time_horizon > 0 else 0
st.markdown(f"**Key Metric:** Compound Annual Growth Rate (CAGR) ≈ {cagr:.1f}%. Bitcoin sweeps drive this—focus on consistent, time-efficient strategies.")

st.markdown("""
**Disclaimer:** Educational simulation; not advice. Markets volatile; taxes/risks apply—consult professionals.
**Suite Tie-In:** Combine with Bitcoin Flywheel for core accumulation, or add leverage from other suite tools for borrowing against swept BTC.
""")

# DGI Integration (example: +10 points for running simulation)
if 'dgi_score' not in st.session_state:
    st.session_state.dgi_score = 50
st.session_state.dgi_score = calculate_dgi(10, st.session_state.dgi_score)
display_dgi()

# PNSD Alignment Expander
with st.expander("North Star Reference"):
    st.markdown("This tool aligns with the Project North Star Document (PNSD) [link to Google Doc]. It supports goals like educating on Bitcoin's compounding and integrating with recommenders for premium-funded flywheels, with DGI for behavioral tracking.")

# Last Update
st.write(f"Last data refresh: {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}")