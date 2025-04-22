import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.analyzer import analyze_sentiment

LABELS = ['Negative 😠', 'Neutral 😐', 'Positive 😊']

st.set_page_config(page_title="SentixAI | Sentiment Dashboard", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
        .stButton>button {
            background-color: #4f46e5;
            color: white;
            font-weight: 600;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #3730a3;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 SentixAI")
st.markdown("### Customer Sentiment Dashboard by MTech.ai")
st.markdown("Easily analyze customer sentiment in real-time or via batch uploads.")

mode = st.radio("Select Mode", ["Single Feedback", "Batch Upload (CSV)"])

if mode == "Single Feedback":
    with st.form("single_form"):
        user_input = st.text_area("Paste feedback or message:", height=160)
        submitted = st.form_submit_button("Analyze Sentiment")

    if submitted and user_input:
        scores, sentiment = analyze_sentiment(user_input)
        st.markdown(f"### ✨ Detected Sentiment: **{sentiment}**")

        fig, ax = plt.subplots()
        bars = ax.bar(LABELS, scores, color=["#e74c3c", "#f1c40f", "#2ecc71"])
        ax.set_ylabel("Confidence Score")
        ax.set_ylim([0, 1])
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + 0.1, yval + 0.01, f"{yval:.2f}")
        st.pyplot(fig)

elif mode == "Batch Upload (CSV)":
    file = st.file_uploader("Upload CSV file with a 'text' column:", type="csv")
    if file:
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column")
        else:
            sentiments = []
            for text in df['text']:
                _, sentiment = analyze_sentiment(str(text))
                sentiments.append(sentiment)
            df['Sentiment'] = sentiments

            st.markdown("#### Sample Results")
            st.dataframe(df.head())

            summary = df['Sentiment'].value_counts()
            st.markdown("#### Sentiment Summary")
            fig2, ax2 = plt.subplots()
            ax2.pie(summary, labels=summary.index, autopct='%1.1f%%',
                    colors=["#e74c3c", "#f1c40f", "#2ecc71"])
            ax2.axis('equal')
            st.pyplot(fig2)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "sentiment_results.csv", "text/csv")

st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2025 MTech.ai | Powered by SentixAI NLP Suite</p>", unsafe_allow_html=True)
