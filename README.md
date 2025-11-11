# Recursive Gene Elimination with Graph Convolutional Networks (RCE-GCN)

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Badges**
     - .. (This section is a placeholder. You can generate and insert your own badges)
       .. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
          :alt: Python Version
       .. image:: https://img.shields.io/badge/streamlit-running-brightgreen.svg
          :alt: Streamlit
       .. image:: https://img.shields.io/badge/code%20style-black-000000.svg
          :alt: code style: black
       .. image:: https://img.shields.io/badge/license-MIT-blue.svg
          :alt: License

   * - **Tests**
     - .. (Placeholder for your CI/CD and code coverage badges)
       .. image:: https://img.shields.io/badge/build-passing-brightgreen.svg
          :alt: Build Status
       .. image:: https://img.shields.io/badge/coverage-0%25-red.svg
          :alt: Code Coverage

   * - **License & DOI**
     - .. (Placeholder for your project's license and Zenodo/DOI badge)
       .. image:: https://img.shields.io/badge/license-MIT-blue.svg
          :alt: License
       .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.YOUR_DOI.svg
          :alt: Zenodo DOI

---

This project provides an interactive web application for cancer classification and the identification of differentially expressed genes from genomic data. It leverages a novel **RCE-GCN (Recursive Gene Elimination with Graph Convolutional Networks)** method, which uses Integrated Gradients to find the most important genetic markers driving disease classification.

The application provides two main workflows:

1.  **Primary RCE-GCN Method**: The main page (`RCE-GCN.py`) runs the full recursive gene elimination pipeline to identify a minimal, high-performance set of genes.
2.  **Alternative GEM-GRAPH Method**: An exploratory, multi-step pipeline (in `pages/Explore Additional Method.py`) that allows users to:
    * Upload and preprocess data.
    * Apply feature selection (Boruta, PCA, etc.).
    * Train a GCN model for classification.
    * Compare GCN performance against baseline models (Random Forest, SVM, MLP, GMM).

## Installation

To install the necessary dependencies, you can clone the repository and use the `requirements.txt` file.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/varshanarayanann/rce-gcn.git](https://github.com/varshanarayanann/rce-gcn.git)
    cd rce-gcn
    ```
   

2.  **Create and activate a virtual environment:**
    * On macOS and Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
   

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
   

## How to Run the Application

Once the installation is complete, you can start the web application with a single command.

1.  Ensure you are in the project's root directory and your virtual environment is activated.

2.  Run the Streamlit app:
    ```bash
    streamlit run RCE-GCN.py
    ```
   

Your web browser will automatically open a new tab with the running application.

## Data Format

To use the application, your input data must be in a **tabular format** (e.g., a CSV file). The data should be structured as follows:

* **Rows**: Each row should represent a different patient or sample.
* **Columns**: Each column should represent a different gene, with the values indicating the expression level.
* A **Label Column** must be present (e.g., 'label') to identify the class for each sample.

**Example:**

| PatientID | Gene_A | Gene_B | Gene_C | ... | label |
| :--- | :--- | :--- | :--- | :-- | :--- |
| Patient_1 | 0.98 | 1.45 | 0.23 | ... | 1 |
| Patient_2 | 1.12 | 0.88 | 0.56 | ... | 0 |
| Patient_3 | 0.45 | 1.99 | 1.04 | ... | 0 |
