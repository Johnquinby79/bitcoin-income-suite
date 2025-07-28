def project_btc_price(start_price=118000.0, start_year=2025, target_year=2045, historical_cagr=0.8220, decay_factor=0.8525):
    """
    Projects Bitcoin end-of-year prices with diminishing returns, rooted in 10-year historical CAGR,
    tapering to exactly reach $10M by 2045. Returns dict {year: price} for suite-wide use,
    enabling efficient projections (e.g., in Opp Cost for hours saved, DCA for monthly sweeps).
    """
    years_ahead = target_year - start_year + 1
    prices = {start_year: start_price}
    P = start_price
    for t in range(1, years_ahead):
        rt = historical_cagr * (decay_factor ** (t - 1))
        P = P * (1 + rt)
        prices[start_year + t] = round(P, 2)  # Clean rounding for tool displays
    return prices