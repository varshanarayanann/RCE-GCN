# pages/2_Explore_Additional_Method.py

import streamlit as st
# Make sure your utility functions are accessible
from utils.introduction import step0_introduction
from utils.upload_csv import step1_upload_display
from utils.feature_selection import step2_feature_selection
from utils.model_call import step3_train_model
from utils.method_comparison import step4_comparison

st.set_page_config(page_title="Explore Additional Methods", layout="wide")

st.title("Explore Additional Method: GEM-GRAPH")
st.markdown("Use the tabs below to proceed through the analysis workflow with other machine learning models.")

# Use tabs for the different steps, which is cleaner than a session state-based menu
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Introduction", 
    "Upload Dataset", 
    "Feature Selection", 
    "Model Training and Results",
    "Method Comparison"
])

with tab1:
    step0_introduction()

with tab2:
    step1_upload_display()

with tab3:
    step2_feature_selection()

with tab4:
    step3_train_model()

with tab5:
    step4_comparison()