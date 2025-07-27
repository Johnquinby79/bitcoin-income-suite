import pandas as pd

def calculate_cagr(data):
    if len(data) < 2:
        return 0.0
    return ((data.iloc[-1] / data.iloc[0]) ** (1 / (len(data) / 252)) - 1) * 100  # Annualized CAGR
