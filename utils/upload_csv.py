import streamlit as st
import pandas as pd

def step1_upload_display():
    st.set_page_config(page_title="GEM-GRAPH App: Upload Data", layout="wide")
    st.title("Step 1: Upload Dataset and Preview")
    uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

    if uploaded_file is not None:
        # Read CSV into pandas DataFrame
        df = pd.read_csv(uploaded_file)

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        df = df.convert_dtypes()

        st.success(f"✅ Successfully loaded dataset with shape {df.shape}")

        # 2. Ask whether to transpose
        transpose_option = st.radio(
            "Do you want to transpose the dataset?",
            ('No', 'Yes')
        )

        if transpose_option == 'Yes':
            df = df.transpose()
            st.info(f"🔄 Transposed dataset. New shape: {df.shape}")

        # 3. Display First 5 Rows Safely
        st.subheader("First 5 Rows of the Dataset")

        if df.shape[1] > 20:
            st.warning("⚠️ Dataset has more than 20 columns. Showing first 20 columns only.")
            st.dataframe(df.iloc[:5, :20])  # First 5 rows and first 20 columns

            with st.expander("Expand to view all columns"):
                st.dataframe(df.head())  # Show full 5 rows if user chooses
        else:
            st.dataframe(df.head())

        # 4. Save the dataframe to Session State for next steps
        st.session_state['uploaded_df'] = df
        st.success(f"Please proceed to the Feature Selection step")

    else:
        st.info("👈 Please upload a .csv file to begin.")