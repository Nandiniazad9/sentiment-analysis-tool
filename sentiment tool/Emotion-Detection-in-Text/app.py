import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from track_utils import (
    create_page_visited_table, 
    add_page_visited_details, 
    view_all_page_visited_details, 
    add_prediction_details, 
    view_all_prediction_details, 
    create_emotionclf_table, 
    IST
)

# Configuration
st.set_page_config(
    page_title="Sentify - Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

pipe_lr = load_model()

# CSS Injection for Premium Look
def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .header-container {
            padding: 3rem 0;
            text-align: center;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 0 0 48px 48px;
            margin-bottom: 3rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .footer-container {
            text-align: center;
            padding: 3rem;
            margin-top: 5rem;
            background: rgba(15, 23, 42, 0.5);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            color: #94a3b8;
        }
        
        h1, h2, h3 {
            color: #f1f5f9 !important;
        }
        
        .prediction-box {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 1rem 0;
            background: linear-gradient(90deg, #6366f1, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 10px 20px rgba(99, 102, 241, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
        <div class="header-container">
            <h1 style='font-size: 3.5rem; margin-bottom: 0; color: #ffffff; text-shadow: 0 4px 12px rgba(0,0,0,0.3);'>🎭 Sentify</h1>
            <p style='font-size: 1.2rem; color: #cbd5e1; font-weight: 500;'>Advanced AI Emotion Analysis</p>
        </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown("""
        <div class="footer-container">
            <p>© 2026 Sentify AI • Built with ❤️ using Streamlit</p>
            <small>Empowering communication through emotional intelligence</small>
        </div>
    """, unsafe_allow_html=True)

# Helper Functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", 
    "happy": "🤗", "joy": "😂", "neutral": "😐", 
    "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Page Handlers
def home_page():
    add_page_visited_details("Home", datetime.now(IST))
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Analyze Your Text")
    
    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("What's on your mind?", placeholder="Type or paste your text here...", height=150)
        submit_text = st.form_submit_button(label='Detect Emotions')
    st.markdown('</div>', unsafe_allow_html=True)

    if submit_text and raw_text:
        col1, col2 = st.columns([1, 1], gap="large")

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)
        add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("### Result")
            emoji_icon = emotions_emoji_dict.get(prediction, "✨")
            st.markdown(f'<div class="prediction-box">{prediction.capitalize()} {emoji_icon}</div>', unsafe_allow_html=True)
            
            # Confidence progress bar
            conf = np.max(probability)
            st.write(f"**Confidence:** {conf:.2%}")
            st.progress(float(conf))
            
            st.info(f"**Original Text:**\n{raw_text}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("### Probability Distribution")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
                x=alt.X('emotions:N', title='Emotions', sort='-y', axis=alt.Axis(labelColor='#94a3b8', titleColor='#f1f5f9')),
                y=alt.Y('probability:Q', title='Probability', axis=alt.Axis(labelColor='#94a3b8', titleColor='#f1f5f9')),
                color=alt.Color('emotions:N', scale=alt.Scale(scheme='magma'), legend=None),
                tooltip=['emotions', 'probability']
            ).properties(height=350, background='transparent').configure_view(
                strokeWidth=0
            ).configure_axis(
                grid=True,
                gridColor='rgba(255,255,255,0.05)'
            )
            st.altair_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

def monitor_page():
    add_page_visited_details("Monitor", datetime.now(IST))
    st.subheader("📊 Analytics Dashboard")

    tab1, tab2 = st.tabs(["Page Metrics", "Prediction Metrics"])

    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("#### Traffic Flow")
            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            chart = alt.Chart(pg_count).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
                x='Page Name', y='Counts', color='Page Name'
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        
        with c2:
            st.write("#### Page Share")
            p = px.pie(pg_count, values='Counts', names='Page Name', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            p.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(p, use_container_width=True)
        
        st.write("#### Raw Access Logs")
        st.dataframe(page_visited_details, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
        
        st.write("#### Emotion Breakdown")
        prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
        pc = alt.Chart(prediction_count).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Counts", type="quantitative"),
            color=alt.Color(field="Prediction", type="nominal"),
            tooltip=['Prediction', 'Counts']
        ).properties(height=400)
        st.altair_chart(pc, use_container_width=True)
        
        st.write("#### Recent Predictions")
        st.dataframe(df_emotions, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def about_page():
    add_page_visited_details("About", datetime.now(IST))
    
    st.markdown("""
<div class="glass-card">
    <h2>Our Mission</h2>
    <p>At <b>Sentify</b>, we bridge the gap between human emotion and digital intelligence. Our application utilizes state-of-the-art NLP models to decode the complex emotional landscape hidden within text.</p>
    
    <div style="margin-top: 2rem;">
        <h3>Key Features</h3>
        <ul>
            <li><b>Real-time Detection:</b> Instant analysis of text inputs.</li>
            <li><b>Confidence Scoring:</b> Detailed probability breakdown for each prediction.</li>
            <li><b>Privacy First:</b> Your data is processed locally and securely.</li>
        </ul>
    </div>
    
    <div style="margin-top: 2rem; background: rgba(99, 102, 241, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #6366f1;">
        <h3>Technical Stack</h3>
        <p>Built with Logistic Regression, Streamlit, and SQLite for robust and efficient performance.</p>
    </div>
</div>
    """, unsafe_allow_html=True)

# Main Application
def main():
    apply_custom_css()
    
    # Sidebar Navigation
    st.sidebar.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("", menu)
    
    # Render Components
    render_header()
    
    create_page_visited_table()
    create_emotionclf_table()
    
    if choice == "Home":
        home_page()
    elif choice == "Monitor":
        monitor_page()
    else:
        about_page()
        
    render_footer()

if __name__ == '__main__':
    main()
