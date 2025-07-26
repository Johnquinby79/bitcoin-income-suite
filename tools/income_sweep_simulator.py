import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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
        prev_btc_value = df.loc[year-1, 'BTC Holdings Value']
        new_btc_value = (prev_btc_value + total_swept) * (1 + btc_growth_rate)

        df.loc[year, 'Portfolio Value'] = new_portfolio
        df.loc[year, 'Annual Dividends'] = dividends
        df.loc[year, 'Swept to BTC (Div)'] = swept_div
        df.loc[year, 'Annual Premiums'] = premiums
        df.loc[year, 'Swept to BTC (Prem)'] = swept_prem
        df.loc[year, 'BTC Holdings Value'] = new_btc_value
        df.loc[year, 'Total Wealth'] = new_portfolio + new_btc_value

    # Create figure for visualization
    fig, ax = plt.subplots()
    ax.plot(df['Year'], df['Total Wealth'], label='Total Wealth')
    ax.plot(df['Year'], df['Portfolio Value'], label='Portfolio (Stocks/ETFs)')
    ax.plot(df['Year'], df['BTC Holdings Value'], label='Bitcoin Holdings')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value ($)')
    ax.set_title(f'Bitcoin Income Sweep Simulation\nStrategy: {strategy_type}')
    ax.legend()
    plt.close(fig)  # Ensures clean Streamlit display

    return df, fig

# Streamlit UI
st.title("Bitcoin Income Sweep Simulator for Beginners")

st.markdown("""
Welcome! This tool shows how you can 'sweep' your dividend income—meaning automatically transfer it—from stocks or ETFs into Bitcoin for long-term savings and growth. Bitcoin has averaged over 50% compound annual growth rate (CAGR) in the last 10 years, far outperforming traditional investments like the S&P 500, making it a powerful income generator when used with discipline.
""")

st.markdown("""
Start with the basics to see how your dividends can grow in Bitcoin over time. For more advanced users, expand below to add extra income from options (like selling contracts on your stocks for premiums), which can boost your results. Remember, the key is staying consistent: Aim to transfer 20-50% of your income to Bitcoin, check your progress every few months, and don't put in more than you can afford.
""")

# Basic inputs
st.markdown("Enter any dollar amount between $100 and $999,999 for your starting investment in dividend-paying stocks or ETFs (e.g., SCHD or VYM). This represents your initial capital that generates dividends to sweep into Bitcoin.")
portfolio_value = st.number_input("Starting Investment Amount ($)", min_value=100, max_value=999999, value=10000, step=100)

dividend_yield = st.slider("Expected Yearly Dividend Payout (%)", 1.0, 100.0, 3.0, 
                           help="How much your stocks or ETFs pay out in dividends each year as a percentage (e.g., 3% means $300 on a $10,000 investment). Some high-performing investments, like MSTY (tied to MicroStrategy's Bitcoin holdings), have delivered over 100% yields due to strong underlying growth. As more companies add Bitcoin to their balance sheets, expect higher payouts—use realistic numbers based on your ETFs or stocks, but start low for conservative planning.") / 100
sweep_percentage = st.slider("Dividend Transfer Percentage to Bitcoin (%)", 0, 100, 50, 
                             help="The portion of your dividends you want to automatically move into Bitcoin for growth.") / 100
time_horizon = st.slider("Savings Timeframe (Years) – The Longer, the Better", 1, 30, 10, 
                         help="How many years you plan to keep sweeping dividends into Bitcoin to let it grow (time is your friend for compounding!).")
stock_growth_rate = st.slider("Expected Yearly Growth of Your Stocks/ETFs (%)", 5.0, 10.0, 7.0, 
                              help="The average annual increase in value of your investments, based on historical market returns (e.g., around 7% for broad stock funds).") / 100
btc_growth_rate = st.slider("Bitcoin Growth Rate (%)", 10.0, 80.0, 15.0, 
                            help="Bitcoin's expected yearly growth rate—adjust based on your view. Over the past 5 years, it averaged about 60% CAGR, but as Bitcoin matures and adoption grows, this may gradually decrease (e.g., to 20-40% in the future). Use conservative estimates for realistic planning.") / 100

# Advanced expander
# Advanced expander
with st.expander("Advanced Users: Integrate Options Premiums"):
    st.markdown("""
    Boost your results with extra income from options strategies (like selling contracts on your stocks or ETFs for premiums). This adds intelligent leverage to your dividends, sweeping the earnings into Bitcoin for faster growth. Stay disciplined—limit options to 20-30% of your portfolio to manage risks like market drops.
    """)
    options_premium_yield = st.slider("Options Premium Income (%)", 0.0, 20.0, 0.0, 
                                      help="The extra yearly income from options as a percentage (e.g., 10% means $1,000 on a $10,000 position).") / 100
    premium_sweep_percentage = st.slider("Premium Transfer Percentage to Bitcoin (%)", 0, 100, 50, 
                                         help="The portion of your options premiums to move into Bitcoin.") / 100
    strategy_type = st.selectbox("Options Strategy Type", ["None", "Covered Calls", "Cash Secured Puts"], 
                                 help="Choose a strategy to simulate—tie it to your suite tools for real ideas.")

    st.text_input("Paste Premium Income (%) from Your Recommender Tool", 
                  help="Enter a value like 12% directly from your Covered Call or Cash Secured Put tool outputs.", 
                  key="pasted_yield")  # Optional: Users can override slider if pasted

    st.markdown("Get premium ideas from suite tools—download and run locally to generate yields (e.g., 10-15% on SPY), then paste the average here—watch how sweeping 60% into Bitcoin accelerates growth by 1.5-3x via intelligent leverage.")
    with open("tools/Covered_Call_Options_Recommender.py", "rb") as file:
        st.download_button(label="Download and Run Covered Call Options Recommender", data=file, file_name="Covered_Call_Options_Recommender.py", help="Download to run: streamlit run this_file.py in CMD")
    with open("tools/Cash_Secured_Put_Recommender.py", "rb") as file:
        st.download_button(label="Download and Run Cash Secured Put Recommender", data=file, file_name="Cash_Secured_Put_Recommender.py", help="Download to run: streamlit run this_file.py in CMD")

    st.subheader("Pros/Cons Table")
    pros_cons = {
        "Aspect": ["Income Boost", "Bitcoin Growth", "Staying Consistent", "Suite Connections"],
        "Pros": ["Adds 5-15% from options on top of dividends for more to transfer.", "Premiums help Bitcoin grow faster over time.", "Builds good habits with automatic transfers.", "Use your recommender tools for real numbers to make simulations better."],
        "Cons": ["Risk of losses if markets drop.", "Taxes on extra income.", "Can feel stressful during ups and downs.", "Takes time to learn the tools."],
        "Mitigation": ["Choose safe options levels; only use what you can afford.", "Use accounts that help with taxes.", "Check every few months and stick to your plan.", "Start with basics, then add recommender data."]
    }
    st.table(pd.DataFrame(pros_cons))

    st.markdown("""
    **Tip for Efficiency:** Input 12% premium from your Cash Secured Put Recommender (e.g., on QQQ), simulate transferring 60% to Bitcoin—see faster growth when combined with dividends for smart wealth building.
    """)

# Simulate and display
df, fig = simulate_income_sweep(portfolio_value, dividend_yield, sweep_percentage, time_horizon,
                                stock_growth_rate, btc_growth_rate, options_premium_yield,
                                premium_sweep_percentage, strategy_type)

st.subheader("Quick Results Summary")
final_wealth = df['Total Wealth'].iloc[-1]
final_btc = df['BTC Holdings Value'].iloc[-1]
st.markdown(f"**Final Total Wealth:** ${final_wealth:.0f} | **Final Final Bitcoin Value:** ${final_btc:.0f}")

st.subheader("Full Projection Table")
# Round to whole dollars and style
df_rounded = df.round(0)
styled_df = df_rounded.style.highlight_max(subset=['Total Wealth', 'BTC Holdings Value'], color='lightgreen').format("{:.0f}")
st.dataframe(styled_df, use_container_width=True)

st.subheader("Growth Chart")
st.pyplot(fig)

cagr = ((df['Total Wealth'].iloc[-1] / df['Total Wealth'].iloc[0]) ** (1 / time_horizon) - 1) * 100 if time_horizon > 0 else 0
st.markdown(f"**Key Result:** Compound Annual Growth Rate (CAGR) ≈ {cagr:.1f}%. Bitcoin transfers drive this—focus on consistent, time-smart plans.")

# Assumptions expander at bottom
with st.expander("View Assumptions & Details"):
    st.markdown("""
    Assumptions: Dividends and premiums are calculated yearly and transferred/reinvested right away; Bitcoin growth is simplified (no daily ups/downs—real markets can vary a lot); taxes or fees aren't included (talk to an expert); past growth rates don't promise future results.
    """)
    st.subheader("Pros/Cons of This Approach")
    strategy_pros_cons = {
        "Aspect": ["Growth Potential", "Risks", "How to Handle"],
        "Details": ["Compounding with Bitcoin's high CAGR (over 50% historically) turns small transfers into big savings over time.", "Bitcoin can drop in value short-term; options add extra risks like losing on trades.", "Spread out your money (e.g., 20-40% to Bitcoin), start small, and check regularly."]
    }
    st.table(pd.DataFrame(strategy_pros_cons))
    st.markdown("For more, check the Bitcoin Flywheel tool assumptions in the suite.")

st.markdown("""
**Disclaimer:** This is for learning only—not financial advice. Markets change; risks and taxes apply—get help from pros.
**Suite Connection:** Build on this with the Bitcoin Flywheel for core basics, or use recommenders for premium boosts—creating efficient income through Bitcoin.
""")