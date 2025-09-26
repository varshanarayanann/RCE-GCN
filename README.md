# Recursive Gene Elimination with Graph Convolutional Networks and Integrated Gradients for Disease Classification in RNA-seq Data

A web-based tool for cancer classification and differentially expressed gene identification from genomic data using the novel **RCE-GCN** method. This application provides an interactive interface for researchers and clinicians to analyze patient data, predict cancer status and type, and understand the key genetic markers driving the prediction.

## Overview

This project provides an interactive web application built with Streamlit that allows users to:
1.  Upload their own tabular genomic data (e.g., gene expression levels for patients).
2.  Analyze the data using our primary **RCE-GCN (Recursive Gene Elimination with Graph Convolutional Networks)** model.
3.  Receive predictions on whether a patient has cancer and the specific type of cancer.
4.  Identify the most important **differentially expressed genes** that the model used to make its classification.
5.  Optionally explore the data using an alternative method, **GEM-GRAPH**, via the sidebar for comparative analysis.

***

## Features

* **Interactive Web Interface**: A user-friendly app built with Streamlit for easy data upload and analysis.
* **Primary RCE-GCN Method**: Leverages a powerful Graph Convolutional Network for high-accuracy predictions.
* **Gene Identification**: Outputs a list of the key differentially expressed genes.
* **Alternative GEM-GRAPH Method**: Includes an additional model for exploratory analysis and comparison.

***

## Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

* Python 3.8 or higher
* `pip` and `venv`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
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
    Make sure your virtual environment is activated, then run:
    ```bash
    pip install -r requirements.txt
    ```

***

## How to Run the Application

Once the installation is complete, you can start the web application with a single command.

1.  Ensure you are in the project's root directory and your virtual environment is activated.

2.  Run the Streamlit app:
    ```bash
    streamlit run RCE-GCN.py
    ```
    Your web browser will automatically open a new tab with the running application.

***

## Data Format

To use the application, your input data must be in a **tabular format** (e.g., a CSV file). The data should be structured as follows:

* **Rows**: Each row should represent a different patient or sample.
* **Columns**: Each column should represent a different gene, with the values indicating the expression level.
* It is recommended to have a column identifying the sample/patient ID.

Here is a small example:

| PatientID | Gene_A | Gene_B | Gene_C | ... |
| :-------- | :----- | :----- | :----- | :-- |
| Patient_1 | 0.98   | 1.45   | 0.23   | ... |
| Patient_2 | 1.12   | 0.88   | 0.56   | ... |
| Patient_3 | 0.45   | 1.99   | 1.04   | ... |
