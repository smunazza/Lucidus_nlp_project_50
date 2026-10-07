import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from urllib.parse import quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Page Config ---
st.set_page_config(
    page_title="Lucidus - Critical Reading Assistant",
    page_icon="LU",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800;900&family=Poppins:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: #1C2333;
}

.stApp {
    background-color: #D3E4DC;
}

.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}

section[data-testid="stSidebar"] {
    background-color: #1C2333;
}
section[data-testid="stSidebar"] * {
    color: #F2F0EA !important;
}

.hero-box {
    background-color: #FFFFFF;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0px 4px 14px rgba(28, 35, 51, 0.08);
}
.app-title {
    font-family: 'Archivo', sans-serif;
    font-size: 52px;
    font-weight: 900;
    color: #1C2333;
    text-transform: uppercase;
    letter-spacing: -1px;
    line-height: 1.1;
}
.app-caption {
    font-size: 15px;
    color: #6B7280;
    margin-top: 6px;
}

div[data-testid="stTextArea"] label p {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #1C2333 !important;
}

textarea {
    border-radius: 10px !important;
    border: 1.5px solid #D8DCE3 !important;
    background-color: #FAFAFA !important;
    color: #1C2333 !important;
}
textarea:focus {
    border: 1.5px solid #F08C6C !important;
}

.stButton button {
    background-color: #F08C6C;
    color: #1C2333;
    border-radius: 8px;
    padding: 10px 26px;
    font-weight: 700;
    font-size: 14px;
    border: none;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stButton button:hover {
    background-color: #E5754F;
    color: #FFFFFF;
}

.stat-card {
    background-color: #FFFFFF;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 14px;
    box-shadow: 0px 4px 14px rgba(28, 35, 51, 0.08);
}
.stat-card-header {
    background-color: #1C2333;
    color: #F2F0EA;
    font-family: 'Archivo', sans-serif;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 10px 18px;
}
.stat-card-body {
    padding: 20px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.stat-number {
    font-family: 'Archivo', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #1C2333;
}
.stat-dot {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 16px;
}
.dot-coral { background-color: #F08C6C; }
.dot-dark { background-color: #1C2333; }
.dot-green { background-color: #6FA98A; }

button[data-baseweb="tab"] {
    font-weight: 700;
    font-size: 14px;
    color: #6B7280 !important;
    text-transform: uppercase;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1C2333 !important;
}

.highlight-box {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 12px;
    line-height: 1.9;
    font-size: 16px;
    box-shadow: 0px 4px 14px rgba(28, 35, 51, 0.08);
    color: #1C2333;
    margin-bottom: 18px;
}

.explanation-box {
    background-color: #FFFFFF;
    border-left: 5px solid #F08C6C;
    padding: 16px 20px;
    border-radius: 8px;
    color: #4B5563;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0px 4px 14px rgba(28, 35, 51, 0.06);
    margin-bottom: 18px;
}

.word-tag {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 4px 6px 4px 0;
}
.tag-strong {
    background-color: #FCE4DD;
    color: #B84E2E;
    border: 1px solid #F08C6C;
}
.tag-weak {
    background-color: #FDF3D9;
    color: #8A6D1D;
    border: 1px solid #E8C55C;
}
</style>
""", unsafe_allow_html=True)

# --- Load NRC Emotion Lexicon ---
nrc = pd.read_csv(
    'NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
    sep='\t', names=['word', 'emotion', 'association']
)
nrc = nrc[nrc['association'] == 1]
loaded_words = set(nrc['word'].unique())
lemmatizer = WordNetLemmatizer()


# --- Load MPQA Subjectivity Lexicon ---
def parse_mpqa_lexicon(filepath):
    mpqa_dict = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            entry = {}
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    entry[key] = value
            word = entry.get('word1')
            subj_type = entry.get('type')
            polarity = entry.get('priorpolarity')
            if word:
                mpqa_dict[word] = {'type': subj_type, 'polarity': polarity}
    return mpqa_dict


mpqa_lexicon = parse_mpqa_lexicon('subjclueslen1-HLTEMNLP05.tff')


@st.cache_resource
def train_clickbait_model():
    clickbait_df = pd.read_csv('clickbait_data.csv')
    X = clickbait_df['headline']
    y = clickbait_df['clickbait']

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_tfidf, y)

    return model, vectorizer


clickbait_model, tfidf_vectorizer = train_clickbait_model()


def predict_clickbait(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = clickbait_model.predict(text_tfidf)[0]
    probability = clickbait_model.predict_proba(text_tfidf)[0][1]
    return bool(prediction), round(probability * 100, 1)

# --- Core Functions ---


def is_confidently_loaded(word):
    lemma = lemmatizer.lemmatize(word.lower(), pos='v')
    in_nrc = lemma in loaded_words or word.lower() in loaded_words
    if not in_nrc:
        return False
    mpqa_entry = mpqa_lexicon.get(word.lower()) or mpqa_lexicon.get(lemma)
    return mpqa_entry is not None


def highlight_loaded_words(text):
    tokens = word_tokenize(text)
    return [word for word in tokens if is_confidently_loaded(word)]


def generate_highlighted_html(text):
    tokens = word_tokenize(text)
    html_parts = []
    for word in tokens:
        if is_confidently_loaded(word):
            html_parts.append(
                f'<span style="background-color: #FFD700; padding: 2px 6px; border-radius: 6px;">{word}</span>')
        else:
            html_parts.append(word)
    return ' '.join(html_parts)


reporting_phrases = [
    "according to", "sources say", "sources claim", "claimed", "alleged",
    "reportedly", "officials said", "experts say", "study shows", "study finds",
    "research shows", "data shows", "survey found", "report states", "cited"
]


def detect_claims(text):
    claims = []
    number_pattern = r'\b\d+(\.\d+)?%?\b'
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        has_number = re.search(number_pattern, sentence)
        has_reporting_phrase = any(phrase in sentence.lower()
                                   for phrase in reporting_phrases)
        if has_number or has_reporting_phrase:
            claims.append(sentence.strip())
    return claims


def generate_verify_link(claim_text):
    query = quote(claim_text)
    return f"https://www.google.com/search?q={query}"


def calculate_density_score(text):
    tokens = word_tokenize(text)
    total_words = len(tokens)
    loaded_count = sum(1 for word in tokens if is_confidently_loaded(word))
    if total_words == 0:
        return 0
    return round((loaded_count / total_words) * 100, 1)


def get_all_loaded_word_details(text):
    tokens = word_tokenize(text)
    results = []
    seen = set()
    for word in tokens:
        if is_confidently_loaded(word) and word.lower() not in seen:
            lemma = lemmatizer.lemmatize(word.lower(), pos='v')
            mpqa_entry = mpqa_lexicon.get(
                word.lower()) or mpqa_lexicon.get(lemma)
            intensity = "Strong" if mpqa_entry['type'] == 'strongsubj' else "Weak"
            results.append({'word': word, 'intensity': intensity})
            seen.add(word.lower())
    return results


# --- Sidebar ---
with st.sidebar:
    st.markdown(
        '<h2 style="color:#FFFFFF; font-family:Archivo; font-size:24px; margin-bottom:5px; letter-spacing:0.5px;">LUCIDUS</h2>'
        '<p style="color:#9AA3B2; font-size:13px; margin-bottom:24px;">Critical Reading Assistant</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="background-color:#2A3245; border-radius:10px; padding:16px 18px; margin-bottom:16px;">'
        '<p style="color:#F08C6C; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">What this tool does</p>'
        '<p style="color:#C6CCD8; font-size:14px; line-height:1.6; margin:0;">'
        'Highlights emotionally loaded language, flags clickbait patterns, and spots factual claims worth verifying.'
        '</p></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="background-color:#2A3245; border-radius:10px; padding:16px 18px; margin-bottom:20px;">'
        '<p style="color:#F08C6C; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">What it does NOT do</p>'
        '<p style="color:#C6CCD8; font-size:14px; line-height:1.6; margin:0;">'
        'It never tells you what is true or false. It helps you think — the judgment stays with you.'
        '</p></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<hr style="border-color:#3A4256; margin-bottom:16px;">'
        '<p style="color:#9AA3B2; font-size:12px; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px;">Project Status</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="display:flex; justify-content:space-between; margin-bottom:6px;">'
        '<span style="color:#C6CCD8; font-size:13px;">Loaded Language</span>'
        '<span style="color:#6FA98A; font-size:13px; font-weight:700;">Done</span></div>'
        '<div style="display:flex; justify-content:space-between; margin-bottom:6px;">'
        '<span style="color:#C6CCD8; font-size:13px;">Clickbait Score</span>'
        '<span style="color:#F08C6C; font-size:13px; font-weight:700;">In Progress</span></div>'
        '<div style="display:flex; justify-content:space-between;">'
        '<span style="color:#C6CCD8; font-size:13px;">Claim Spotter</span>'
        '<span style="color:#6B7280; font-size:13px; font-weight:700;">Pending</span></div>',
        unsafe_allow_html=True
    )

# --- Header ---
st.markdown(
    '<div class="hero-box">'
    '<div class="app-title">Lucidus</div>'
    '<div class="app-caption">Illuminating the language of persuasion, reading beyond the rhetoric.</div>'
    '</div>',
    unsafe_allow_html=True
)

# --- Input ---
user_text = st.text_area("Paste your article or headline here:", height=200)
analyze_button = st.button("Analyze")

if analyze_button and user_text:
    tab1, tab2, tab3 = st.tabs(
        ["Loaded Language", "Clickbait Score", "Claim Spotter"])

    with tab1:
        html_output = generate_highlighted_html(user_text)
        density = calculate_density_score(user_text)
        loaded_count = len(highlight_loaded_words(user_text))
        total_words = len(word_tokenize(user_text))
        word_details = get_all_loaded_word_details(user_text)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-header">Emotional Density</div>
                <div class="stat-card-body">
                    <div class="stat-number">{density}%</div>
                    <div class="stat-dot dot-coral">%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-header">Loaded Words</div>
                <div class="stat-card-body">
                    <div class="stat-number">{loaded_count}</div>
                    <div class="stat-dot dot-dark">#</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-header">Total Words</div>
                <div class="stat-card-body">
                    <div class="stat-number">{total_words}</div>
                    <div class="stat-dot dot-green">W</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="highlight-box">{html_output}</div>', unsafe_allow_html=True)

        # Word-level intensity tags
        if word_details:
            tags_html = ""
            for d in word_details:
                tag_class = "tag-strong" if d['intensity'] == "Strong" else "tag-weak"
                tags_html += f'<span class="word-tag {tag_class}">{d["word"]} — {d["intensity"]}</span>'
            st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="explanation-box">'
            'The highlighted words are chosen for emotional impact, not just information. Ask yourself: '
            'what facts would remain if these words were replaced with neutral ones? Strong language can '
            'make a claim feel more urgent or certain than the evidence actually supports.'
            '</div>',
            unsafe_allow_html=True
        )

    with tab2:
        is_clickbait, confidence = predict_clickbait(user_text)

        col1, col2 = st.columns(2)
        with col1:
            verdict = "Likely Clickbait" if is_clickbait else "Likely Not Clickbait"
            dot_class = "dot-coral" if is_clickbait else "dot-green"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-header">Verdict</div>
                <div class="stat-card-body">
                    <div class="stat-number" style="font-size:22px;">{verdict}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-header">Confidence</div>
                <div class="stat-card-body">
                    <div class="stat-number">{confidence}%</div>
                    <div class="stat-dot dot-dark">%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="explanation-box">'
            'This score is based on patterns learned from thousands of labeled headlines — words, phrasing, '
            'and structure commonly seen in clickbait versus straightforward reporting. It is a statistical '
            'estimate, not a judgment on whether the content itself is accurate.'
            '</div>',
            unsafe_allow_html=True
        )

    with tab3:
        claims = detect_claims(user_text)

        if claims:
            st.markdown(
                f"<p style='color:#1C2333; font-size:15px; margin-bottom:16px;'>Found {len(claims)} claim(s) worth verifying:</p>", unsafe_allow_html=True)

            for i, claim in enumerate(claims, 1):
                link = generate_verify_link(claim)
                st.markdown(f"""
                <div class="highlight-box" style="margin-bottom:12px;">
                    <p style="margin:0 0 10px 0;">{claim}</p>
                    <a href="{link}" target="_blank" style="color:#F08C6C; font-weight:600; font-size:14px; text-decoration:none;">Verify this claim →</a>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(
                '<div class="explanation-box">'
                'These sentences contain numbers, statistics, or attribution phrases that typically '
                'signal factual claims. This tool does not verify them for you — click through to check '
                'the claim against a trusted source yourself.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("No clear factual claims detected in this text.")
