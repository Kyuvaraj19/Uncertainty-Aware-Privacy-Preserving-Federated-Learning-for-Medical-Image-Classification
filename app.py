# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

st.set_page_config(
    page_title="Federated Learning Dashboard",
    layout="wide"
)

st.title("🏥 Uncertainty-Aware Federated Learning")

DATA_FILE = "training_logs.csv"

placeholder = st.empty()

while True:

    if os.path.exists(DATA_FILE):

        df = pd.read_csv(DATA_FILE)

        with placeholder.container():

            st.subheader("Training Metrics")

            col1, col2, col3 = st.columns(3)

            # Latest Metrics
            col1.metric(
                "Latest Loss",
                f"{df['loss'].iloc[-1]:.4f}"
            )

            col2.metric(
                "Latest Accuracy",
                f"{df['accuracy'].iloc[-1]:.4f}"
            )

            col3.metric(
                "Latest Uncertainty",
                f"{df['uncertainty'].iloc[-1]:.4f}"
            )

            # LOSS GRAPH
            fig1, ax1 = plt.subplots()

            ax1.plot(df["round"], df["loss"])

            ax1.set_title("Loss vs Federated Round")

            ax1.set_xlabel("Round")

            ax1.set_ylabel("Loss")

            st.pyplot(fig1)

            # ACCURACY GRAPH
            fig2, ax2 = plt.subplots()

            ax2.plot(df["round"], df["accuracy"])

            ax2.set_title("Accuracy vs Federated Round")

            ax2.set_xlabel("Round")

            ax2.set_ylabel("Accuracy")

            st.pyplot(fig2)

            # UNCERTAINTY GRAPH
            fig3, ax3 = plt.subplots()

            ax3.plot(df["round"], df["uncertainty"])

            ax3.set_title("Uncertainty vs Federated Round")

            ax3.set_xlabel("Round")

            ax3.set_ylabel("Uncertainty")

            st.pyplot(fig3)

            st.dataframe(df)

    else:
        st.warning("Waiting for training logs...")

    time.sleep(2)