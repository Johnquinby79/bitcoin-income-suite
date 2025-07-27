import streamlit as st

def calculate_dgi(action_points=0, base_score=50):
    """Simple DGI calculation: Base 50 + points from actions (e.g., +10 for delayed choice)."""
    dgi_score = base_score + action_points
    return min(100, max(0, dgi_score))

def display_dgi():
    if 'dgi_score' not in st.session_state:
        st.session_state.dgi_score = 50
    st.metric("Delayed Gratification Index (DGI)", st.session_state.dgi_score, help="Your score for shifting to delayed gratification—higher means better long-term habits for Bitcoin wealth.")
