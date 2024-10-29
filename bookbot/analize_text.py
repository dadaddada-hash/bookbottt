# Import necessary libraries
import re
import nltk
from textblob import TextBlob
import spacy
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
from collections import Counter
import textstat  # Importing textstat for readability metrics

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to analyze a text file
def analyze_text(file_path):
    # Read and process file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Debugging: Raw text output
    print(f"--- Raw Text ---\n{text}\n")  # Debugging-Zeile
    if not text.strip():  # Überprüfen, ob der Text leer ist
        print("Die Datei enthält keinen Text.")
        return

    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Debugging: Tokenized output
    print(f"--- Tokenized Sentences ---\n{sentences}\n")  # Debugging-Zeile
    print(f"--- Tokenized Words ---\n{words}\n")          # Debugging-Zeile

    word_count = len(words)
    sentence_count = len(sentences)

    # 1. Basic Text Statistics
    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
    else:
        avg_word_length = 0  # Default to 0 if no words are found

    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # Debugging: Basic statistics output
    print(f"--- Basic Text Statistics ---")
    print(f"Word Count: {word_count}")
    print(f"Sentence Count: {sentence_count}")
    print(f"Average Word Length: {avg_word_length:.2f}")
    print(f"Average Sentence Length: {avg_sentence_length:.2f}\n")

    # 2. Vocabulary Analysis
    fdist = FreqDist(words)
    unique_words = len(fdist.keys())
    most_common_words = fdist.most_common(10)
    vocabulary_richness = unique_words / word_count if word_count > 0 else 0

    # Debugging: Vocabulary analysis output
    print(f"--- Vocabulary Analysis ---")
    print(f"Unique Words: {unique_words}")
    print(f"Most Common Words: {most_common_words}")
    print(f"Vocabulary Richness (Type-Token Ratio): {vocabulary_richness:.2f}\n")

    # 3. Sentiment Analysis
    blob = TextBlob(text)
    overall_sentiment = blob.sentiment.polarity
    sentence_sentiments = [TextBlob(sentence).sentiment.polarity for sentence in sentences]

    # Debugging: Sentiment analysis output
    print(f"--- Sentiment Analysis ---")
    print(f"Overall Sentiment (Polarity): {overall_sentiment:.2f}")

    # Plotting sentiment over time
    plt.plot(sentence_sentiments)
    plt.title("Sentiment Shifts Over Time")
    plt.xlabel("Sentence Number")
    plt.ylabel("Sentiment Polarity")
    plt.show()

    # 4. Character & Keyword Analysis
    doc = nlp(text)
    characters = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    character_freq = Counter(characters)

    # Debugging: Character analysis output
    print(f"--- Character & Keyword Analysis ---")
    print(f"Most Frequent Characters: {character_freq.most_common(5)}\n")

    # 5. Readability Metrics using textstat
    flesch = textstat.flesch_reading_ease(text)
    gunning_fog = textstat.gunning_fog(text)

    # Debugging: Readability metrics output
    print(f"--- Readability Metrics ---")
    print(f"Flesch-Kincaid Reading Ease: {flesch:.2f}")
    print(f"Gunning Fog Index: {gunning_fog:.2f}\n")

    # 6. Narrative Analysis (Dialogue vs. Narrative)
    dialogue_count = len(re.findall(r'“[^”]*”', text))
    narrative_count = len(sentences) - dialogue_count

    # Debugging: Narrative analysis output
    print(f"--- Narrative Analysis ---")
    print(f"Dialogue Count: {dialogue_count}")
    print(f"Narrative Count: {narrative_count}")
    dialogue_to_narrative_ratio = (dialogue_count / narrative_count) if narrative_count > 0 else 0
    print(f"Dialogue vs. Narrative Ratio: {dialogue_to_narrative_ratio:.2f}")

# Run analysis
file_path = r'C:\Users\Huda Arain\OneDrive\Desktop\hello bookbot\bookbot\hello.txt'
analyze_text(file_path)
