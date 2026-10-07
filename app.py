from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.parse import quote
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Load NRC Emotion Lexicon ---
nrc = pd.read_csv(
    'C:/Users/sydsh/OneDrive/Documents/Lucidus/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
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
                mpqa_dict[word] = {
                    'type': subj_type,
                    'polarity': polarity
                }
    return mpqa_dict


mpqa_lexicon = parse_mpqa_lexicon('subjclueslen1-HLTEMNLP05.tff')


# --- Core Helper: Confidence Check ---
def is_confidently_loaded(word):
    """Check karta hai ki word NRC mein hai AND MPQA mein confident match (Strong/Weak) bhi hai"""
    lemma = lemmatizer.lemmatize(word.lower(), pos='v')

    in_nrc = lemma in loaded_words or word.lower() in loaded_words
    if not in_nrc:
        return False

    mpqa_entry = mpqa_lexicon.get(word.lower()) or mpqa_lexicon.get(lemma)

    return mpqa_entry is not None


# --- Core Functions ---
def highlight_loaded_words(text):
    tokens = word_tokenize(text)
    found = [word for word in tokens if is_confidently_loaded(word)]
    return found


def generate_highlighted_html(text):
    tokens = word_tokenize(text)
    html_parts = []
    for word in tokens:
        if is_confidently_loaded(word):
            html_parts.append(
                f'<span style="background-color: #FFD700; padding: 2px; border-radius: 3px;">{word}</span>')
        else:
            html_parts.append(word)
    return ' '.join(html_parts)


def calculate_density_score(text):
    tokens = word_tokenize(text)
    total_words = len(tokens)

    loaded_count = sum(1 for word in tokens if is_confidently_loaded(word))

    if total_words == 0:
        return 0

    density = (loaded_count / total_words) * 100
    return round(density, 1)


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


# --- Tests ---
text = "The shocking scandal has devastated the community, leaving officials furious and residents terrified."

result = highlight_loaded_words(text)
print("Loaded words found:", result)

html_result = generate_highlighted_html(text)
print(html_result)

density = calculate_density_score(text)
print("Density score:", density)

print("Total loaded words in NRC lexicon:", len(loaded_words))
print("Total words in MPQA lexicon:", len(mpqa_lexicon))

details = get_all_loaded_word_details(text)
print("\nWord-level details:")
for d in details:
    print(d)


print("\n--- Debugging 'saddening' ---")
print("Lemma:", lemmatizer.lemmatize("saddening", pos='v'))
print("In NRC?", lemmatizer.lemmatize("saddening", pos='v')
      in loaded_words or "saddening" in loaded_words)
print("In MPQA?", mpqa_lexicon.get("saddening") or mpqa_lexicon.get(
    lemmatizer.lemmatize("saddening", pos='v')))


# --- Claim Spotter ---
reporting_phrases = [
    "according to", "sources say", "sources claim", "claimed", "alleged",
    "reportedly", "officials said", "experts say", "study shows", "study finds",
    "research shows", "data shows", "survey found", "report states", "cited"
]


def detect_claims(text):
    claims = []

    # 1. Numbers / percentages / statistics
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


test_text = "According to officials, unemployment rose to 15% last month. The weather was nice today."
claims = detect_claims(test_text)
for c in claims:
    print(c)
    print(generate_verify_link(c))
    print()

clickbait_df = pd.read_csv('clickbait_data.csv')
print(clickbait_df.head())
print(clickbait_df.columns)
print(clickbait_df.shape)


# --- Clickbait Classifier ---
X = clickbait_df['headline']
y = clickbait_df['clickbait']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

clickbait_model = LogisticRegression()
clickbait_model.fit(X_train_tfidf, y_train)

y_pred = clickbait_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Clickbait model accuracy:", accuracy)

# Test with a new headline


def predict_clickbait(headline):
    headline_tfidf = tfidf_vectorizer.transform([headline])
    prediction = clickbait_model.predict(headline_tfidf)[0]
    probability = clickbait_model.predict_proba(headline_tfidf)[0][1]
    return prediction, probability


test_headline = "You Won't Believe What This Celebrity Did Next"
pred, prob = predict_clickbait(test_headline)
print(f"\nHeadline: {test_headline}")
print(f"Clickbait: {bool(pred)}, Confidence: {round(prob*100, 1)}%")

test_headline2 = "Senate Passes New Infrastructure Bill After Months of Negotiation"
pred2, prob2 = predict_clickbait(test_headline2)
print(f"\nHeadline: {test_headline2}")
print(f"Clickbait: {bool(pred2)}, Confidence: {round(prob2*100, 1)}%")
