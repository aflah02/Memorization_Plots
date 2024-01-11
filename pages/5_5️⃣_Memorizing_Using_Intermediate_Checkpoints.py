import streamlit as st
import pandas as pd
import plotly
import plotly.express as px
import matplotlib.pyplot as plt

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

    str_types = ['non_random_strings', 'random_strings']
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    ckpts = [0,1,4,16,64,256,1000, 2000, 10000, 50000, 100000, 143000]
    # for i in range(2000, 144000, 1000):
    #     ckpts.append(i)
    
    str_type = 'non_random_strings'
    for size in sizes:
        col_names = [get_col_name(str_type, size, ckpt) for ckpt in ckpts]
        subset_df = df[col_names]
        # plot
        fig = plt.figure()

        data = subset_df.values

        # plot each column
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=subset_df.columns[i].split(" ")[1])

        # set x and y axis labels
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        # title
        plt.title(f"Loss for {str_type} of size {size}")

        # change legend title to Layer
        plt.legend(title='Layer')

        # reduce legend size
        plt.legend(prop={'size': 6})

        st.pyplot(fig)

    str_type = 'random_strings'
    for size in sizes:
        col_names = [get_col_name(str_type, size, ckpt) for ckpt in ckpts]
        subset_df = df[col_names]
        # plot
        fig = plt.figure()

        data = subset_df.values

        # plot each column
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=subset_df.columns[i].split(" ")[1])

        # set x and y axis labels
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        # title
        plt.title(f"Loss for {str_type} of size {size}")

        # change legend title to Layer
        plt.legend(title='Layer')

        # reduce legend size
        plt.legend(prop={'size': 6})

        st.pyplot(fig)

if __name__ == "__main__":
    main()