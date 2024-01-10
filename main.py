import streamlit as st
import os

# Get the list of images
image_options = os.listdir('accuracy_plots_peft_methods/')

st.title('Accuracy Plots for PEFT Methods')

string_lens = [16,32,64,128,256,512,1024]
methods = ['full-finetuning', 'ia3', 'lora', 'p-tuning', 'prefix-tuning', 'prompt-tuning']
suffixes = ['', 'Target modules- q_proj,v_proj Feed forward modules- down_proj', 
            'r- 8 alpha- 32 dropout- 0.1', 'Encoder hidden size- 128 Num virtual tokens- 8',
            'Num virtual tokens- 8 Init text- This is a test']
string_types = ['random_strings', 'non_random_strings']

chosen_string_len = st.multiselect('Choose string length', string_lens)
chosen_method = st.multiselect('Choose method', methods)
chosen_type = st.multiselect('Choose string type', string_types)

suffixes_for_chosen_methods = []
for method in chosen_method:
    suffixes_for_chosen_methods.append(suffixes[methods.index(method)])

def build_file_name(string_len, method, string_type, suffix):
    base = 'accuracy_plots_peft_methods/accuracy_' + string_type + ' pythia-1b ' + method + ' ' + str(string_len) + 'x1 Epochs- 100 BSize- 1 Seed- 42'
    if suffix != '':
        base += ' ' + suffix
    base += '.png'
    return base

# sort string lengths by first converting to int
chosen_string_len = sorted(chosen_string_len, key=lambda x: int(x))

for method, suffix in zip(chosen_method, suffixes_for_chosen_methods):
    for string_type in chosen_type:
        for string_len in chosen_string_len:
            file_name = build_file_name(string_len, method, string_type, suffix)
            title = file_name
            # remove accuracy_plots_fixed/accuracy_ from start
            title = title[len('accuracy_plots_fixed/accuracy_'):]
            # remove .png from end
            title = title[:-len('.png')]
            st.markdown('#### ' + title)
            st.image(file_name, use_column_width=True)
            

