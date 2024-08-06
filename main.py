from deep_translator import GoogleTranslator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd

# Download resource VADER
nltk.download("vader_lexicon")

print("\n\n")


x = pd.read_csv("data/data.csv", sep=",", encoding="latin1")

sample = x.head()

for i in sample["Review"].values.tolist():
    # Text dalam bahasa Indonesia
    text = i

    # Terjemahkan teks ke bahasa Inggris
    translator = GoogleTranslator(source="id", target="en")
    translated_text = translator.translate(text)

    # Analisis sentimen menggunakan VADER
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(translated_text)

    print(text)
    # Cetak skor sentimen
    print(f"Negative: {sentiment_scores['neg']}")
    print(f"Neutral: {sentiment_scores['neu']}")
    print(f"Positive: {sentiment_scores['pos']}")
    print(f"Compound: {sentiment_scores['compound']}")

    # Interpretasi skor compound
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positif"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negatif"
    else:
        sentiment = "Netral"

    print(f"Sentimen keseluruhan: {sentiment}")
    print()
    print()
