import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
from datetime import datetime
import sys
sys.path.append('.')
from utils.dgi import calculate_dgi, display_dgi  # Shared DGI util

# Fetch current BTC price (fallback to hardcoded if API fails)
try:
    btc_data = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd').json()
    btc_initial_price = btc_data['bitcoin']['usd']
except:
    btc_initial_price = 118378  # Fallback based on July 24, 2025 data

# Assumptions (conservative default mean return)
years_default = 5
btc_annual_mean_return_default = 30  # Conservative default in %
btc_volatility = 0.5
dividend_yields = {'MSTY': 1.30, 'WNTR': 0.25, 'Custom': 0.50}
buffer_annual_return = 0.02
num_simulations = 20

# Initialize session state for last simulation
if 'last_sim' not in st.session_state:
    st.session_state['last_sim'] = None

# Helper: Simulate BTC price with volatility
def get_btc_price(year, random_returns):
    return btc_initial_price * np.prod(1 + random_returns[:year + 1])

# Generalized simulation function (runs one scenario)
def simulate_scenario(initial_investment, years, scenario, selected_assets, allocations, ltv_percent, borrowing_apr, include_buffer, buffer_percent, btc_mean_return, apply_diminishing):
    # Compute weighted dividend yield if multiple assets (for passive/hybrid)
    weighted_yield = 0
    if scenario in ['passive_only', 'hybrid']:
        if selected_assets:
            weighted_yield = sum((allocations[asset] / 100) * dividend_yields[asset] for asset in selected_assets)

    # Adjust for diminishing returns if enabled
    mean_returns = np.full(years + 1, btc_mean_return / 100)
    if apply_diminishing:
        for y in range(1, years + 1):
            mean_returns[y] *= (0.95 ** y)  # 5% relative reduction per year

    results = []
    for _ in range(num_simulations):
        random_returns = np.random.normal(mean_returns, btc_volatility)
        df = pd.DataFrame(columns=['Year', 'Wealth_USD', 'BTC_Equivalent', 'Buffer_Used', 'Cumulative_Interest', 'Cumulative_Loan_Balance'])
        principal = initial_investment if scenario == 'passive_only' else initial_investment / 2 if scenario == 'hybrid' else 0
        btc_holdings = 0
        debt = 0
        buffer = initial_investment * buffer_percent if include_buffer else 0
        buffer_used_total = 0
        cum_interest = 0
        cum_loan_balance = 0

        leverage_ratio = 1 / (1 - ltv_percent / 100) if ltv_percent > 0 and scenario in ['leverage_only', 'hybrid'] else 1

        if scenario in ['leverage_only', 'hybrid']:
            lev_part = initial_investment if scenario == 'leverage_only' else initial_investment / 2
            debt = lev_part * (leverage_ratio - 1)
            btc_holdings += (lev_part * leverage_ratio) / btc_initial_price

        for year in range(years + 1):
            btc_price = get_btc_price(year, random_returns)
            net_income = 0
            yearly_interest = 0

            if year > 0:
                if scenario in ['passive_only', 'hybrid']:
                    dividend = principal * weighted_yield
                    net_income += dividend
                    btc_holdings += dividend / btc_price  # Reinvest to BTC

                if scenario in ['leverage_only', 'hybrid']:
                    btc_holdings *= (1 + random_returns[year])
                    yearly_interest = debt * borrowing_apr
                    net_income -= yearly_interest
                    debt += yearly_interest  # Compound debt
                    cum_interest += yearly_interest
                    cum_loan_balance = debt  # End balance

                if include_buffer and net_income < 0:
                    cover_amount = min(-net_income, buffer)
                    buffer -= cover_amount
                    buffer_used_total += cover_amount
                    net_income += cover_amount

                if include_buffer:
                    buffer *= (1 + buffer_annual_return)

                if net_income > 0:
                    btc_holdings += net_income / btc_price

            wealth_usd = (btc_holdings * btc_price) + principal + buffer - debt
            btc_equiv = wealth_usd / btc_price
            df.loc[year] = [year, round(wealth_usd, 2), round(btc_equiv, 6), round(buffer_used_total, 2), round(cum_interest, 2), round(cum_loan_balance, 2)]

        results.append(df)

    avg_df = pd.concat(results).groupby(level=0).mean()
    return avg_df

# Streamlit UI
st.title("BTC Flywheel Simulator: Grow Wealth with Bitcoin")
st.markdown("See how Bitcoin generates income over time using ETFs, dividends, borrowing, and smart planning.")
st.write(f"Current Bitcoin Price: ${btc_initial_price:,.2f}")

initial_investment = st.number_input("How much money are you starting with? (max $999,999)", min_value=100, max_value=999999, value=1000, step=1,
                                     help="Enter the USD for your first BTC buy. If you already own BTC, use its current USD value here.")

years = st.slider("How many years do you want to simulate?", 3, 20, years_default,
                  help="This strategy works best over long periods. Short times can show big losses from BTC's ups and downs, but that's what drives strong long-term gains.")

btc_mean_return = st.slider("Expected BTC Annual Return %:", 10, 100, btc_annual_mean_return_default, step=1,
                            help="Average yearly BTC growth (conservative: 20-40%; historical higher but future may slow).")

apply_diminishing = st.checkbox("Apply Diminishing Returns? (For realistic long-term growth)", value=False,
                                help="Reduces growth rate over time to model BTC's maturing market.")

scenario = st.selectbox("Strategy: Pick your approach.", ["Leverage Only", "Passive Only", "Hybrid - Passive + Leverage"],
                        help="Leverage Only: Borrow to buy more BTC (bigger gains, bigger risks). Passive Only: Use ETF dividends to buy BTC. Hybrid - Passive + Leverage: Mix for balance.")

# Dynamic hiding based on scenario
show_dividends = scenario != "Leverage Only"
show_leverage = scenario != "Passive Only"

selected_assets = []
allocations = {}
if show_dividends:
    selected_assets = st.multiselect("Dividend Asset(s) (for Passive/Hybrid):", list(dividend_yields.keys()),
                                     help="Pick one or more ETFs for dividends. Allocate % below (must total 100%). Helps spread risk in up or down markets.")
    for asset in selected_assets:
        if asset == 'Custom':
            dividend_yields['Custom'] = st.number_input("Custom Yield % (e.g., 50 for 50%):", 0.0, 200.0, 50.0, step=1.0) / 100
        allocations[asset] = st.number_input(f"% Allocation for {asset}:", 0, 100, 50 if len(selected_assets) == 1 else 0, step=1)

    total_alloc = sum(allocations.values())
    if selected_assets and total_alloc != 100:
        st.error("Allocations must add up to 100%.")

ltv_percent = 20
borrowing_apr = 12.0
if show_leverage:
    ltv_percent = st.slider("Loan-to-Value (LTV) %: How much to borrow against your BTC.", 1, 30, 20, step=1,
                            help="LTV is the borrowed portion of your position (e.g., 20% means borrow $200 on $1,000 for $1,200 total). Higher LTV boosts rewards but raises risk of losses.")
    borrowing_apr = st.slider("Borrowing APR %: Interest rate on loans.", 5.0, 20.0, 12.0, step=0.1,
                              help="Cost of borrowing in % (e.g., 12.00%). Lower rates help your BTC grow faster.")

include_buffer = st.checkbox("Include Buffer Savings? (Optional: Cover costs in down markets without selling.)",
                             help="Sets aside cash to handle tough times, keeping your BTC intact.")

buffer_percent = 20.0
if include_buffer:
    buffer_percent = st.slider("Buffer %: Safe cash allocation.", 10, 50, 20, step=1,
                               help="% of starting money set aside as cash. Earns low return but protects during drops.")

if st.button("Run Simulation"):
    if show_dividends and selected_assets and total_alloc != 100:
        st.error("Fix allocations to 100% before running.")
    else:
        with st.spinner("Running simulation..."):
            current_df = simulate_scenario(initial_investment, years, scenario.lower().replace(' - passive + leverage', '').replace(' ', '_'), selected_assets, allocations, ltv_percent if show_leverage else 0, borrowing_apr / 100 if show_leverage else 0, include_buffer, buffer_percent / 100, btc_mean_return, apply_diminishing)

        # Check for previous simulation and prepare comparison
        has_comparison = st.session_state['last_sim'] is not None
        if has_comparison:
            last_scenario = st.session_state['last_sim']['scenario']
            last_df = st.session_state['last_sim']['df']
            st.markdown(f"Comparing Your Simulations: Last run ({last_scenario}) vs. current ({scenario}). See how different strategies add dividends or leveraged gains into BTC over time. Run another to update—focus on long-term discipline for best results!")

        st.subheader("Results: Average Over Simulations (With Market Ups/Downs)")
        st.markdown("Shows growth in USD and BTC. Yields based on last 12 months (can vary).")

        if has_comparison:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Previous: {last_scenario}**")
                st.dataframe(last_df)
            with col2:
                st.write(f"**Current: {scenario}**")
                st.dataframe(current_df)
        else:
            st.dataframe(current_df)

        # Comparison Chart
        fig, ax = plt.subplots()
        if has_comparison:
            ax.plot(last_df['Year'], last_df['BTC_Equivalent'], label=f'Previous: {last_scenario}', linestyle='--')
        ax.plot(current_df['Year'], current_df['BTC_Equivalent'], label=f'Current: {scenario}')
        ax.set_xlabel('Years')
        ax.set_ylabel('Average Wealth in BTC')
        ax.set_title('Your Growth in BTC Terms')
        ax.legend()
        st.pyplot(fig)

        if include_buffer:
            buffer_used = current_df['Buffer_Used'].iloc[-1]
            st.markdown(f"Buffer covered ~${buffer_used:.2f} in down periods, letting income add into BTC.")

        st.markdown("**Note:** Past data used; real results vary. Leverage can lose money—use discipline. Consult an advisor.")

        # Auto-save current as last for next run
        st.session_state['last_sim'] = {'scenario': scenario, 'df': current_df}

        # DGI Integration (example: +10 points for running simulation)
        if 'dgi_score' not in st.session_state:
            st.session_state.dgi_score = 50
        st.session_state.dgi_score = calculate_dgi(10, st.session_state.dgi_score)
        display_dgi()

        # Export for Suite
        final_value = current_df['Wealth_USD'].iloc[-1]
        annualized_yield = ((final_value / initial_investment) ** (1 / years) - 1) * 100 if years > 0 else 0
        export_text = f"{annualized_yield:.2f}"
        st.text_area("Copy this Annualized Yield for Income Sweep Simulator", export_text, height=50)

        # PNSD Alignment Expander
        with st.expander("North Star Reference"):
            st.markdown("This tool aligns with the Project North Star Document (PNSD) [link to Google Doc]. It supports goals like educating on Bitcoin's compounding and integrating with recommenders for premium-funded flywheels, with DGI for behavioral tracking.")

# Last Update
st.write(f"Last data refresh: {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}")