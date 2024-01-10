import streamlit as st
import os

st.set_page_config(
    page_title="Main",
    page_icon="🎯",
)

st.title('Plots for Memorization Experiments')

st.markdown("This is a Streamlit dashboard for visualizing the results of the memorization experiments.")
st.markdown("Different pages (accessible from the sidebar) contain results for different experiments.")