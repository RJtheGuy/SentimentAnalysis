import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.analyzer import analyze_sentiment
import asyncio
import nest_asyncio

nest_asyncio.apply()

st.set_page_config(
    page_title="SentixAI | Sentiment Dashboard",
    page_icon="🤖",
    layout="centered"
)

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
        .stAlert {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 SentixAI")
st.markdown("### Customer Sentiment Dashboard")
st.markdown("Analyze customer sentiment in real-time or via batch uploads.")

LABELS = ['Negative 😠', 'Neutral 😐', 'Positive 😊']

mode = st.radio(
    "Select Mode",
    ["Single Feedback", "Batch Upload (CSV)"],
    horizontal=True
)

if mode == "Single Feedback":
    with st.form("single_form"):
        user_input = st.text_area(
            "Paste feedback or message:",
            height=160,
            placeholder="Enter your text here..."
        )
        submitted = st.form_submit_button("Analyze Sentiment")

    if submitted:
        if not user_input.strip():
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner("Analyzing sentiment..."):
                try:
                    scores, sentiment = analyze_sentiment(user_input)
                    
                    # Display results
                    st.markdown(f"### ✨ Detected Sentiment: **{sentiment}**")
                    
                    # Visualization
                    fig, ax = plt.subplots()
                    colors = ["#e74c3c", "#f1c40f", "#2ecc71"]
                    bars = ax.bar(LABELS, scores, color=colors)
                    
                    # Customize plot
                    ax.set_ylabel("Confidence Score")
                    ax.set_ylim([0, 1])
                    for bar in bars:
                        yval = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2,
                            yval + 0.02,
                            f"{yval:.2f}",
                            ha='center',
                            va='bottom'
                        )
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Please check your internet connection and try again")

elif mode == "Batch Upload (CSV)":
    st.info("Ensure your CSV has a column named 'text' containing the feedback")
    file = st.file_uploader(
        "Upload CSV file:",
        type="csv",
        accept_multiple_files=False
    )
    
    if file:
        try:
            df = pd.read_csv(file)
            
            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column")
            else:
                with st.spinner("Analyzing batch data..."):
                    # Process in batches for large files
                    batch_size = 10
                    results = []
                    
                    for i in range(0, len(df), batch_size):
                        batch = df['text'].iloc[i:i+batch_size]
                        batch_results = [
                            analyze_sentiment(str(text)) 
                            for text in batch
                        ]
                        results.extend(batch_results)
                    
                    # Add results to dataframe
                    df['Scores'] = [r[0] for r in results]
                    df['Sentiment'] = [r[1] for r in results]
                    
                    # Show sample results
                    st.markdown("#### Sample Results")
                    st.dataframe(df.head())
                    
                    # Summary visualization
                    st.markdown("#### Sentiment Distribution")
                    fig2, ax2 = plt.subplots()
                    summary = df['Sentiment'].value_counts()
                    ax2.pie(
                        summary,
                        labels=summary.index,
                        autopct='%1.1f%%',
                        colors=["#e74c3c", "#f1c40f", "#2ecc71"],
                        startangle=90
                    )
                    ax2.axis('equal')
                    st.pyplot(fig2)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Full Results",
                        csv,
                        "sentiment_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>© 2025 SentixAI | Powered by MTech.ai</p>", 
    unsafe_allow_html=True
)