import streamlit as st
import pandas as pd
import plotly
import plotly.express as px

st.set_page_config(page_title="LoRA Hyperparameter Tuning", page_icon="3️⃣")

def main():
    st.title("Loss Curves")
    st.markdown("Loss curves for different hyperparameter settings can be found [here](https://api.wandb.ai/links/aflah/5aj7ns1x)")
    
if __name__ == "__main__":
    main()