import streamlit as st
import pandas as pd
import plotly
import plotly.express as px

st.set_page_config(page_title="Memorizing New String On Intermediate Checkpoint from memorizing an old string", page_icon="4️⃣")

def main():
    st.title("Loss Curves")
    st.markdown("Loss curves when starting from different intermediate checkpoints can be found [here](https://api.wandb.ai/links/aflah/y80ctll8)")
    
if __name__ == "__main__":
    main()