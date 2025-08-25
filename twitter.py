import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random, io
from datetime import datetime, timedelta
from collections import Counter

# ==============================
# Demo Data Generator
# ==============================
def generate_demo_tweets(num_tweets=1000):
    brands = ["Tesla", "Apple", "Google", "Amazon", "Microsoft", "Netflix", "Spotify", "Nike", "Adidas", "Coca-Cola"]
    sentiments = [
        "Absolutely love {brand}'s latest innovation! Mind-blowing! 🚀🔥",
        "{brand} products are a total rip-off. So disappointed! 😡💔",
        "Just unboxed my new {brand} device – pure joy! 😍✨",
        "{brand} app crashes every time. Fix it already! 😤",
        "Thrilled with {brand}'s service! Highly recommend! 👍🌟",
        "{brand} is killing it with their updates! 📈🎉",
        "Why is {brand} so unreliable? Constant issues. 🙄",
        "Mixed feelings about {brand}. It's alright, I guess. 🤷‍♂️",
        "{brand} changed the game forever! Legendary! 👑",
        "Wasted money on {brand}. Never buying again! 💸🚫"
    ]

    tweets = []
    for _ in range(num_tweets):
        brand = random.choice(brands)
        text = random.choice(sentiments).format(brand=brand)
        created_at = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        tweets.append({"created_at": created_at, "text": text})
    return pd.DataFrame(tweets)

# Initialize session
if 'tweets_df' not in st.session_state:
    st.session_state.tweets_df = generate_demo_tweets(1200)
if 'analysis_df' not in st.session_state:
    st.session_state.analysis_df = pd.DataFrame()

# ==============================
# Page Config & Styling
# ==============================
st.set_page_config(page_title="🚀 Sentiment Tracker Pro", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a, #1e293b); color: #f1f5f9; }
    h1,h2,h3,h4 { color: #38bdf8; }
    .stButton>button { background: #38bdf8; color: black; font-weight: bold; border-radius: 8px; transition: 0.3s; }
    .stButton>button:hover { background: #0ea5e9; color: white; }
    .css-1cpxqw2 { background: #22c55e; }
    .stTextInput>div>div>input { border: 2px solid #38bdf8; border-radius: 8px; }
    .stDataFrame { border: 1px solid #334155; border-radius: 8px; }
    .metric-card { background:#1e293b; padding:15px; border-radius:10px; text-align:center; box-shadow:0 0 10px rgba(0,0,0,.4);} 
    </style>
""", unsafe_allow_html=True)

# ==============================
# Sidebar Controls
# ==============================
with st.sidebar:
    st.header("⚙️ Controls")
    keyword = st.text_input("🔍 Brand Keyword", "Tesla")
    tweet_count = st.slider("Tweets to Load", 50, 500, 150)
    sentiment_threshold = st.slider("Sentiment Threshold", 0.0, 0.5, 0.1, step=0.05)
    auto_refresh = st.checkbox("🔄 Auto Refresh Every Load", value=False)
    include_wordcloud = st.checkbox("Show Word Cloud", value=True)
    include_radar = st.checkbox("Show Radar Sentiment", value=True)
    
    st.markdown("---")
    custom_tweet = st.text_area("✍️ Add Custom Tweet")
    if st.button("Add Tweet") and custom_tweet:
        st.session_state.tweets_df = pd.concat([st.session_state.tweets_df, pd.DataFrame([{ "created_at": datetime.now(), "text": custom_tweet }])], ignore_index=True)
        st.success("Tweet added!")

# ==============================
# Load & Analyze
# ==============================
if st.button("🚀 Run Analysis") or auto_refresh:
    df = st.session_state.tweets_df
    df = df[df['text'].str.contains(keyword, case=False)].head(tweet_count)
    if df.empty:
        st.error("No tweets found for this keyword!")
    else:
        def analyze(row):
            blob = TextBlob(row['text'])
            pol = blob.sentiment.polarity
            sent = "Positive" if pol > sentiment_threshold else "Negative" if pol < -sentiment_threshold else "Neutral"
            return pd.Series([pol, sent])
        
        df[['Polarity','Sentiment']] = df.apply(analyze, axis=1)
        df['Time'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('Time')
        st.session_state.analysis_df = df

# ==============================
# Main Content
# ==============================
if not st.session_state.analysis_df.empty:
    df = st.session_state.analysis_df
    total = len(df)
    avg_pol = df['Polarity'].mean()
    counts = df['Sentiment'].value_counts()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tweets", total)
    col2.metric("Avg Polarity", f"{avg_pol:.2f}")
    col3.metric("Positive", counts.get('Positive',0))
    col4.metric("Negative", counts.get('Negative',0))

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Insights", "📥 Export"])

    # Dashboard
    with tab1:
        fig_pie = px.pie(df, names="Sentiment", title="Sentiment Distribution", hole=0.4,
                         color="Sentiment", color_discrete_map={"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#facc15"})
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_line = px.line(df, x="Time", y="Polarity", title="Sentiment Over Time", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

        if include_radar:
            radar_vals = [counts.get('Positive',0), counts.get('Negative',0), counts.get('Neutral',0)]
            radar = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=["Positive","Negative","Neutral"], fill='toself'))
            radar.update_layout(title="Radar Sentiment Profile", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(radar, use_container_width=True)

    # Insights
    with tab2:
        if include_wordcloud:
            text_all = ' '.join(df['text'])
            wc = WordCloud(width=800,height=400,background_color='black',colormap='cool').generate(text_all)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        st.subheader("🔥 Trending Words")
        words = [w for t in df['text'] for w in t.split() if not w.startswith("http")]
        freq = Counter(words).most_common(10)
        st.bar_chart(pd.DataFrame(freq, columns=["Word","Count"]).set_index("Word"))

        st.subheader("🏆 Top Positive")
        st.write(df[df['Sentiment']=="Positive"].sort_values('Polarity',ascending=False).head(5)[['Time','text','Polarity']])

        st.subheader("⚠️ Top Negative")
        st.write(df[df['Sentiment']=="Negative"].sort_values('Polarity').head(5)[['Time','text','Polarity']])

    # Export
    with tab3:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download CSV", csv, "sentiment.csv","text/csv")

        json = df.to_json(orient="records").encode('utf-8')
        st.download_button("🗂 Download JSON", json, "sentiment.json","application/json")

        buf = io.BytesIO()
        fig_pie.write_image(buf, format="png")
        st.download_button("📊 Download Pie Chart", buf, "chart.png","image/png")

# Footer
st.markdown("---")
st.markdown("<center>✨ Sentiment Tracker Pro | Neon UI | Interactive | 2025 🚀</center>", unsafe_allow_html=True)
