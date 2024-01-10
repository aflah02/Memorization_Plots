import streamlit as st
import pandas as pd
import plotly
import plotly.express as px

st.set_page_config(page_title="Memorizing Using Intermediate Checkpoints", page_icon="5️⃣")

def get_col_name(str_type, size, ckpt):
    return f"{str_type} pythia-1b_{ckpt} full-finetuning {size}x1 Epochs- 100 BSize- 1 Seed- 42"

def main():
    st.title("Loss Comparison")

    if 'page_5_data' not in st.session_state:
        df = pd.read_csv("data/losses_memorization_intermediate_checkpoints.csv")
        st.session_state.page_5_data = df
    else:
        df = st.session_state.page_5_data

    str_types = ['random_strings']
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    ckpts = [0,1,4,16,64,256,1000, 2000, 10000, 50000, 100000, 143000]
    # for i in range(2000, 144000, 1000):
    #     ckpts.append(i)
    
    for str_type in str_types:
        for size in sizes:
            col_names = [get_col_name(str_type, size, ckpt) for ckpt in ckpts]
            subset_df = df[col_names]
            # plot
            fig = px.line(subset_df, y=col_names, x=subset_df.index)
            # set x and y axis labels
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Loss",
            )
            # change name in legend
            fig.for_each_trace(lambda t: t.update(name=t.name.split(" ")[1]))

            # title
            fig.update_layout(title=f"Loss for {str_type} of size {size}")

            # change legend title to Layer
            fig.update_layout(legend_title_text='Layer')
            st.plotly_chart(fig)




    # cols = df.columns.tolist()
    # # col name must contain unfreeze
    # cols = [x for x in cols if "unfreeze" in x]
    # df = df[cols]
    # # remove cols with loss__MAX or loss__MIN in name
    # cols = [x for x in cols if "MAX" not in x and "MIN" not in x]
    # df = df[cols]
    
    # # remove  - eval/loss from end of col name
    # cols = [x.replace(" - eval/loss", "") for x in cols]
    # df.columns = cols
    # # split colname on - and keep -1
    # cols = [x.split("-")[-1] for x in cols]
    # df.columns = cols
    
    # # sort cols by int value
    # cols = sorted(cols, key=lambda x: int(x))
    # df = df[cols]

    # # get 16 random colors
    # all_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10 + px.colors.qualitative.T10 + px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    # colors = all_colors[:len(cols)]

    # mapping = {}
    # for i, col in enumerate(cols):
    #     mapping[col] = colors[i]

    # select_custom = st.checkbox("Select Custom")
    # if select_custom:
    #     selected = st.multiselect("Select Runs", cols, default=cols)
    # else:
    #     selected = []
    #     select_first_k = st.checkbox("Select first k")
    #     select_last_k = st.checkbox("Select last k")
    #     if select_first_k:
    #         k = st.slider("Select first k", 1, len(cols), 3)
    #         selected.extend(cols[:k])
    #     if select_last_k:
    #         k = st.slider("Select last k", 1, len(cols), 3)
    #         selected.extend(cols[-k:])
    #     selected = list(set(selected))
    #     selected = sorted(selected, key=lambda x: int(x))



    # colors_selected = [mapping[x] for x in selected]
    # # sort selected by int value
    # selected = sorted(selected, key=lambda x: int(x))
    # # plot line chart
    # fig = px.line(df, y=selected, x=df.index, color_discrete_map=mapping)
    # # set x and y axis labels
    # fig.update_layout(
    #     xaxis_title="Epoch",
    #     yaxis_title="Loss",
    # )
    # # change legend title to Layer
    # fig.update_layout(legend_title_text='Layer')
    # st.plotly_chart(fig)
    
if __name__ == "__main__":
    main()