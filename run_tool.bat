@echo off
cd C:\Users\johnq\bitcoin_income_suite
call ..\bitcoin_suite_env\Scripts\activate.bat
streamlit run tools\income_sweep_simulator.py