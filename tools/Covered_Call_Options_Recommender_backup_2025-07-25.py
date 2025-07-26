import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
import math
import pytz

# Friday holidays 2025 (full closures on Fridays)
friday_holidays = [datetime(2025, 4, 18).date(), datetime(2025, 7, 4).date()]

# Black-Scholes functions for Greeks
def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price, d1, d2

def calculate_greeks(S, K, T, r, sigma):
    call_price, d1, d2 = black_scholes_call(S, K, T, r, sigma)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)
    return delta, gamma, theta / 365

# Function to calculate RSI (14-period)
def calculate_rsi(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1mo')['Close']
    if len(hist) < 15:
        return 50  # Neutral if insufficient data
    delta = hist.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# Function to calculate Bollinger Bands (20-period, 2SD)
def calculate_bollinger(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1mo')['Close']
    sma = hist.rolling(20).mean().iloc[-1]
    std = hist.rolling(20).std().iloc[-1]
    upper = sma + 2 * std
    lower = sma - 2 * std
    current = hist.iloc[-1]
    return current > upper, current < lower

# Function to calculate MACD (12,26,9)
def calculate_macd(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1mo')['Close']
    ema12 = hist.ewm(span=12, adjust=False).mean()
    ema26 = hist.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1] > signal.iloc[-1]  # Bullish if True

# Function to calculate Stochastic Oscillator (14-period)
def calculate_stochastic(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1mo')
    low14 = hist['Low'].rolling(14).min()
    high14 = hist['High'].rolling(14).max()
    k = 100 * (hist['Close'] - low14) / (high14 - low14)
    return k.iloc[-1]

# Function to calculate ATR (14-period)
def calculate_atr(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1mo')
    high_low = hist['High'] - hist['Low']
    high_close = (hist['High'] - hist['Close'].shift()).abs()
    low_close = (hist['Low'] - hist['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]
    return atr

# Function to calculate Put-Call Ratio (from chain, simplified volume ratio)
def calculate_pcr(ticker, exp):
    stock = yf.Ticker(ticker)
    calls = stock.option_chain(exp).calls['volume'].sum()
    puts = stock.option_chain(exp).puts['volume'].sum()
    return puts / calls if calls > 0 else 1

# Function to calculate simple Volume Profile (high-volume price from 1mo hist)
def calculate_volume_profile(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1mo')
    hist['price_bin'] = hist['Close'].round(0)  # Bin by integer price
    vol_profile = hist.groupby('price_bin')['Volume'].sum().idxmax()
    return vol_profile

# App Title
st.title("Covered Call Recommendation Tool")

st.markdown("""
Enter a stock ticker and number of contracts to get recommendations for covered calls.
Recommendations are based on optimal metrics for premium vs risk balance.
Click 'Calculate Recommendations' to fetch current data and generate suggestions.
Filters adjust dynamically (up to 5 levels) if no results, with risk tolerance rank (1 optimal/low risk - 5 loosest/higher risk).
""")

# Initialize session state
if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []
if 'last_fetch' not in st.session_state:
    st.session_state.last_fetch = "Not fetched yet"
if 'current_price' not in st.session_state:
    st.session_state.current_price = None
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None

# Sidebar Inputs
st.sidebar.header("Recently Viewed")
st.sidebar.markdown("<p style='font-size:small;'>Select a ticker to view</p>", unsafe_allow_html=True)

# Stacked recent searches as buttons
if st.session_state.recent_searches:
    for tick in st.session_state.recent_searches:
        if st.sidebar.button(tick, key=f"recent_{tick}"):
            st.session_state.ticker_input = tick

ticker_input = st.sidebar.text_input("Stock Ticker (e.g., TSLA)", value=st.session_state.get('ticker_input', ''))
contracts = st.sidebar.number_input("Number of Contracts", min_value=1, value=1)

# Sidebar notification
st.sidebar.markdown("---")
fetch_time = st.session_state.last_fetch if st.session_state.last_fetch else datetime.now().strftime('%m/%d/%Y %H:%M:%S')
st.sidebar.info(f"Last API call: {fetch_time}")

# Last update time
st.write(f"Last data refresh: {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}")

# Assumptions (base)
risk_free_rate = 0.04
max_dte = 45

# Filter tiers (1 strictest/optimal to 5 loosest) - Lowered max_delta for lower assignment prob
filter_tiers = [
    {'min_delta': 0.10, 'max_delta': 0.25, 'max_gamma': 0.03, 'min_oi': 200, 'min_volume': 100, 'min_otm_pct': 7, 'max_otm_pct': 12, 'min_dte': 14},
    {'min_delta': 0.10, 'max_delta': 0.30, 'max_gamma': 0.04, 'min_oi': 150, 'min_volume': 75, 'min_otm_pct': 5, 'max_otm_pct': 15, 'min_dte': 10},
    {'min_delta': 0.10, 'max_delta': 0.35, 'max_gamma': 0.05, 'min_oi': 100, 'min_volume': 50, 'min_otm_pct': 3, 'max_otm_pct': 18, 'min_dte': 7},
    {'min_delta': 0.05, 'max_delta': 0.40, 'max_gamma': 0.06, 'min_oi': 50, 'min_volume': 25, 'min_otm_pct': 2, 'max_otm_pct': 20, 'min_dte': 4},
    {'min_delta': 0.05, 'max_delta': 0.45, 'max_gamma': 0.07, 'min_oi': 10, 'min_volume': 10, 'min_otm_pct': 0, 'max_otm_pct': 25, 'min_dte': 1}
]

# Function to get target expiration dates
def get_target_exps():
    today = datetime.now().date()
    days_to_friday = (4 - today.weekday()) % 7
    this_friday = today + timedelta(days=days_to_friday)
    if this_friday in friday_holidays:
        this_target = this_friday - timedelta(days=1)
    else:
        this_target = this_friday
    
    two_fridays = this_friday + timedelta(weeks=2)
    if two_fridays in friday_holidays:
        two_target = two_fridays - timedelta(days=1)
    else:
        two_target = two_fridays
    
    four_fridays = this_friday + timedelta(weeks=4)
    if four_fridays in friday_holidays:
        four_target = four_fridays - timedelta(days=1)
    else:
        four_target = four_fridays
    
    # Format as 'YYYY-MM-DD'
    targets = [
        this_target.strftime('%Y-%m-%d'),
        two_target.strftime('%Y-%m-%d'),
        four_target.strftime('%Y-%m-%d')
    ]
    return targets

# Function to assign risk rating (1-5 based on delta)
def get_risk_rating(delta, level):
    base_rating = 1 if delta < 0.1 else 2 if delta < 0.2 else 3 if delta < 0.3 else 4 if delta < 0.4 else 5
    weighted_rating = base_rating + (level - 1)  # Weight up if looser tier
    return min(5, weighted_rating)  # Cap at 5

# Function to fetch and recommend with dynamic filters
def get_recommendations(ticker, contracts):
    stock = yf.Ticker(ticker)
    current_price = stock.info.get('regularMarketPrice', stock.info.get('previousClose', 0))
    rsi = calculate_rsi(ticker)
    bb_upper, bb_lower = calculate_bollinger(ticker)
    macd_bullish = calculate_macd(ticker)
    stochastic = calculate_stochastic(ticker)
    atr = calculate_atr(ticker)
    vol_profile = calculate_volume_profile(ticker)
    exps = stock.options
    if not exps:
        return pd.DataFrame(), None, rsi, bb_upper, bb_lower, macd_bullish, stochastic, atr, vol_profile, current_price
    
    target_exps = get_target_exps()
    # Find available exps matching or closest to targets
    available_targets = []
    for target in target_exps:
        if target in exps:
            available_targets.append(target)
        else:
            # Find closest >= today
            closest = min((exp for exp in exps if exp >= target), key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(target, '%Y-%m-%d')), default=None)
            if closest:
                available_targets.append(closest)
    
    all_candidates = pd.DataFrame()
    for exp in available_targets:
        chain = stock.option_chain(exp).calls
        chain['mid'] = (chain['bid'] + chain['ask']) / 2
        chain['yield'] = chain['mid'] / current_price * 100
        chain['otm_pct'] = (chain['strike'] - current_price) / current_price * 100
        chain['impliedVolatility'] = chain['impliedVolatility'].fillna(0)
        chain['pcr'] = calculate_pcr(ticker, exp)
        T = max((datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days / 365, 0.001)
        def apply_greeks(row):
            if row['impliedVolatility'] > 0:
                return calculate_greeks(current_price, row['strike'], T, risk_free_rate, row['impliedVolatility'])
            return 0, 0, 0
        chain['delta'], chain['gamma'], chain['theta'] = zip(*chain.apply(apply_greeks, axis=1))
        chain['dte'] = (datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days
        chain['expiration'] = exp
        all_candidates = pd.concat([all_candidates, chain])
    
    if all_candidates.empty:
        return pd.DataFrame(), None, rsi, bb_upper, bb_lower, macd_bullish, stochastic, atr, vol_profile, current_price
    
    # Adjust filters based on RSI
    if rsi > 70:  # Overbought: Favor higher delta for premium
        for tier in filter_tiers:
            tier['max_delta'] += 0.05  # Increase risk tolerance slightly
    elif rsi < 30:  # Oversold: Lower delta to reduce assignment
        for tier in filter_tiers:
            tier['max_delta'] -= 0.05  # Decrease risk tolerance

    # Try tiers until results found
    for level, filters in enumerate(filter_tiers, 1):
        candidates = all_candidates[
            (all_candidates['dte'] >= filters['min_dte']) &
            (all_candidates['otm_pct'].between(filters['min_otm_pct'], filters['max_otm_pct'])) &
            (all_candidates['delta'].between(filters['min_delta'], filters['max_delta'])) &
            (all_candidates['gamma'] < filters['max_gamma']) &
            (all_candidates['openInterest'] > filters['min_oi']) &
            (all_candidates['volume'] > filters['min_volume'])
        ]
        if not candidates.empty:
            # RSI-adjusted score
            rsi_factor = 1 + (rsi / 100 - 0.5) if rsi > 70 else 1 / (1 + (0.5 - rsi / 100)) if rsi < 30 else 1
            # Additional indicator factors
            bb_factor = 1.2 if bb_upper else 0.8 if bb_lower else 1
            macd_factor = 0.9 if macd_bullish else 1.1  # Favor bearish for calls
            stoch_factor = 1.2 if stochastic > 80 else 0.8 if stochastic < 20 else 1
            atr_factor = 1 + (atr / current_price) if atr > current_price * 0.02 else 1  # Boost for volatile
            pcr_factor = 1.1 if candidates['pcr'].mean() > 1 else 0.9  # Bearish sentiment good for calls
            vp_factor = 1.1 if abs(candidates['strike'].mean() - vol_profile) < atr else 1
            candidates['score'] = (candidates['yield'] * -candidates['theta']) / (candidates['delta'] * candidates['gamma'] + 1e-6) * rsi_factor * bb_factor * macd_factor * stoch_factor * atr_factor * pcr_factor * vp_factor
            candidates['risk_rating'] = candidates['delta'].apply(lambda d: get_risk_rating(d, level))
            candidates['premium_per_contract'] = candidates['mid'] * 100
            candidates['total_premium'] = candidates['premium_per_contract'] * contracts
            candidates['premium_per_contract_fmt'] = candidates['premium_per_contract'].apply(lambda x: f"${x:,.2f}")
            candidates['total_premium_fmt'] = candidates['total_premium'].apply(lambda x: f"${x:,.2f}")
            candidates['strike_fmt'] = candidates['strike'].apply(lambda x: f"${x:,.2f}")
            candidates['expiration_fmt'] = candidates['expiration'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%m/%d/%Y'))
            # Sort per group by delta
            grouped = candidates.groupby('expiration_fmt').apply(lambda g: g.sort_values('delta').nlargest(3, 'score'))
            # Overall top
            overall_top = candidates.nlargest(1, 'score')
            if len(overall_top) > 1:
                overall_top = overall_top.nlargest(1, 'premium_per_contract')
                if len(overall_top) > 1:
                    overall_top = overall_top.nsmallest(1, 'dte')
            return grouped, level, overall_top, rsi, bb_upper, bb_lower, macd_bullish, stochastic, atr, vol_profile, current_price
    
    return pd.DataFrame(), None, pd.DataFrame(), rsi, bb_upper, bb_lower, macd_bullish, stochastic, atr, vol_profile, current_price

# Main Display
if st.button("Calculate Recommendations"):
    if ticker_input:
        # Update recent searches (last 5)
        if ticker_input.upper() not in st.session_state.recent_searches:
            st.session_state.recent_searches.insert(0, ticker_input.upper())
            st.session_state.recent_searches = st.session_state.recent_searches[:5]
        
        st.session_state.last_fetch = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
        
        st.header(f"Recommendations for {ticker_input.upper()} (Least Risk/Low Premium to High Risk/High Premium)")
        grouped_rec, risk_tolerance, overall_top, rsi, bb_upper, bb_lower, macd_bullish, stochastic, atr, vol_profile, current_price = get_recommendations(ticker_input.upper(), contracts)
        st.session_state.current_price = current_price
        st.session_state.current_ticker = ticker_input.upper()
        if not grouped_rec.empty:
            # Display grouped
            for exp, group in grouped_rec.groupby(level=0):
                st.subheader(f"{exp} Options")
                display_df = group[['expiration_fmt', 'strike_fmt', 'premium_per_contract_fmt', 'total_premium_fmt', 'risk_rating']]
                display_df.columns = ['Expiration', 'Strike', 'Premium Per Contract', 'Total Premium', 'Risk Rating']
                st.dataframe(display_df)
            
            st.write(f"Premium per contract is the income for one option; total is for {contracts} contracts.")
            st.metric("Risk Tolerance Rank", risk_tolerance, help="1: Most optimal/low risk; 5: Loosest/higher risk")
            
            # Most optimal
            st.header("Most Optimal Option")
            optimal_df = overall_top[['expiration_fmt', 'strike_fmt', 'premium_per_contract_fmt', 'total_premium_fmt', 'risk_rating']]
            optimal_df.columns = ['Expiration', 'Strike', 'Premium Per Contract', 'Total Premium', 'Risk Rating']
            # Highlight green
            def highlight_optimal(s):
                return ['background-color: lightgreen' for _ in s]
            st.dataframe(optimal_df.style.apply(highlight_optimal, axis=1))
            
            # Explanation
            optimal = overall_top.iloc[0]
            rsi_note = f"RSI at {rsi:.2f}: " + ("Overbought, favoring higher premium with managed risk." if rsi > 70 else ("Oversold, reducing assignment risk during potential rebound." if rsi < 30 else "Neutral, standard balance."))
            bb_note = "Bollinger Bands: " + ("Upper hit, boosting premium score for overbought." if bb_upper else ("Lower hit, de-emphasizing risk for oversold." if bb_lower else "Neutral."))
            macd_note = "MACD: " + ("Bullish, slightly de-emphasizing calls." if macd_bullish else "Bearish, favoring calls for premium.")
            stoch_note = f"Stochastic at {stochastic:.2f}: " + ("Overbought (>80), boosting premium." if stochastic > 80 else ("Oversold (<20), reducing risk." if stochastic < 20 else "Neutral."))
            atr_note = f"ATR {atr:.2f}: " + ("High volatility, boosting score for premium potential." if atr > current_price * 0.02 else "Normal.")
            vp_note = f"Volume Profile high at {vol_profile:.2f}: " + ("Strike near support, boosting score." if abs(optimal['strike'] - vol_profile) < atr else "Neutral.")
            st.write(f"**Why this is optimal:** Highest score ({optimal['score']:.2f}) balancing yield ({optimal['yield']:.2f}%) and theta ({optimal['theta']:.2f}) against delta ({optimal['delta']:.2f}, risk {optimal['risk_rating']}) and gamma ({optimal['gamma']:.2f}). Premium: {optimal['premium_per_contract_fmt']}. DTE: {optimal['dte']} days. {rsi_note} {bb_note} {macd_note} {stoch_note} {atr_note} {vp_note}")

            # New: Cumulative Yield Calculation and Export
            st.header("Estimated Annualized Premium Yield")
            if not grouped_rec.empty:
                avg_premium_per_contract = grouped_rec['premium_per_contract'].mean()
                portfolio_value = current_price * 100  # Assume standard contract size for yield %
                weekly_premium = avg_premium_per_contract * contracts
                annual_premium = weekly_premium * 50  # Assume 50 weeks/year for rolling trades
                annualized_yield = (annual_premium / portfolio_value) * 100
                st.write(f"Based on average premium from recommendations (${avg_premium_per_contract:.2f} per contract), assuming weekly rolls:")
                st.metric("Estimated Annualized Yield", f"{annualized_yield:.2f}%", help="This is a conservative estimate; actual depends on trade execution and market conditions. Use in Income Sweep Simulator for Bitcoin sweeps.")
                
                # Copyable text for export
                export_text = f"{annualized_yield:.2f}"
                st.text_area("Copy this Yield for Income Sweep Simulator", export_text, height=50)
            else:
                st.write("No recommendations available for yield calculation.")

            # New: Expandable Footer for Risk Metrics
            with st.expander("Understand Risk Metrics"):
                st.markdown("""
                **Risk Rating (1-5 Scale):** This evaluates the likelihood of the option being exercised (assignment risk) based on delta. 1 = Lowest risk (e.g., delta <0.1, less chance of assignment, safer for premium collection). 5 = Highest risk (e.g., delta >0.4, higher assignment chance but potentially higher premiums). Use lower ratings for conservative strategies when sweeping premiums with ETF dividends into Bitcoin.

                **Risk Tolerance Rank (1-5):** This shows how the tool adjusted filters to find options. 1 = Strictest criteria (e.g., high liquidity, low delta) for safer, optimal trades. 5 = Loosest (e.g., lower volume, higher delta) for more options but increased risk. Lower ranks align with suite discipline for efficient premium generation to allocate to Bitcoin sweeps (e.g., 20-30% portfolio exposure to options).

                These metrics help with intelligent leverage—use low-risk options to generate steady premiums, sweep them with ETF dividends into Bitcoin for long-term wealth, maintaining allocations and discipline (quarterly reviews).
                """)
        else:
            st.warning("No suitable recommendations found even after loosening filters.")
    else:
        st.error("Please enter a stock ticker.")

# Last update time
st.write(f"Last data refresh: {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}")