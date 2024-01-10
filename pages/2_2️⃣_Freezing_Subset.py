import streamlit as st
import pandas as pd
import plotly
import plotly.express as px

st.set_page_config(page_title="Freezing Subset", page_icon="2️⃣")

def main():
    st.title("Entropy Plots")
    st.markdown("Entropy results can be found [here](https://docs.google.com/document/d/1W9Jwf5W0QrhMjYLA18KP5ZCwkMsElVVFhwQD-Es3g34/edit?usp=sharing)")
    st.title("Correctness Plots")
    st.markdown("Correctness results can be found [here](https://docs.google.com/document/d/1jLW62VSt_Z9n4HdpfVypa183nptfvKrOHwNzfrH4b0U/edit?usp=sharing)")
    st.title("Loss Comparison")
    df = pd.read_csv("data/loss_data_freezing_plots.csv")
    cols = df.columns.tolist()
    # col name must contain unfreeze
    cols = [x for x in cols if "unfreeze" in x]
    df = df[cols]
    # remove cols with loss__MAX or loss__MIN in name
    cols = [x for x in cols if "MAX" not in x and "MIN" not in x]
    df = df[cols]
    
    # remove  - eval/loss from end of col name
    cols = [x.replace(" - eval/loss", "") for x in cols]
    df.columns = cols
    # split colname on - and keep -1
    cols = [x.split("-")[-1] for x in cols]
    df.columns = cols
    
    # sort cols by int value
    cols = sorted(cols, key=lambda x: int(x))
    df = df[cols]

    # get 16 random colors
    all_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10 + px.colors.qualitative.T10 + px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    colors = all_colors[:len(cols)]

    mapping = {}
    for i, col in enumerate(cols):
        mapping[col] = colors[i]

    select_custom = st.checkbox("Select Custom")
    if select_custom:
        selected = st.multiselect("Select Runs", cols, default=cols)
    else:
        selected = []
        select_first_k = st.checkbox("Select first k")
        select_last_k = st.checkbox("Select last k")
        if select_first_k:
            k = st.slider("Select first k", 1, len(cols), 3)
            selected.extend(cols[:k])
        if select_last_k:
            k = st.slider("Select last k", 1, len(cols), 3)
            selected.extend(cols[-k:])
        selected = list(set(selected))
        selected = sorted(selected, key=lambda x: int(x))



    colors_selected = [mapping[x] for x in selected]
    # sort selected by int value
    selected = sorted(selected, key=lambda x: int(x))
    # plot line chart
    fig = px.line(df, y=selected, x=df.index, color_discrete_map=mapping)
    # set x and y axis labels
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
    )
    # change legend title to Layer
    fig.update_layout(legend_title_text='Layer')
    st.plotly_chart(fig)
    
if __name__ == "__main__":
    main()