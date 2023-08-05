import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from bs4 import BeautifulSoup
import re

# Load the input text file and read its contents
with open('C:/Users/Adavelli Rohan Reddy/Desktop/VIT/6th sem/RPA/Project/Email_Auto_modern/Email/input.txt', 'r', encoding='utf-8-sig') as file:
    text = file.read()

# Remove HTML tags from the text using BeautifulSoup
soup = BeautifulSoup(text, 'html.parser')
text = soup.get_text()

# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Define patterns to filter out unwanted sentences
patterns = [r'^https?://', r'^www\.', r'^\w+@\w+\.\w+']

# Filter out unwanted sentences
filtered_sentences = []
for sent in sentences:
    if not any(re.match(pattern, sent) for pattern in patterns):
        filtered_sentences.append(sent)

# Tokenize the filtered sentences into words
words = word_tokenize(' '.join(filtered_sentences))

# Remove stop words from the text
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.casefold() not in stop_words]

# Perform stemming on the words
ps = PorterStemmer()
stemmed_words = [ps.stem(word) for word in filtered_words]

# Create a frequency distribution of the stemmed words
word_frequencies = FreqDist(stemmed_words)

# Calculate the weighted frequency of each sentence
weighted_frequencies = {}
for i, sent in enumerate(filtered_sentences):
    for word in word_tokenize(sent.lower()):
        if ps.stem(word) in word_frequencies:
            if i in weighted_frequencies:
                weighted_frequencies[i] += word_frequencies[ps.stem(word)]
            else:
                weighted_frequencies[i] = word_frequencies[ps.stem(word)]

# Calculate the average sentence score
total_sentences = len(weighted_frequencies)
average_score = sum(weighted_frequencies.values()) / total_sentences

# Build a graph of the sentences based on cosine similarity
sentence_graph = np.zeros((total_sentences, total_sentences))
for i in range(total_sentences):
    for j in range(total_sentences):
        if i != j:
            sentence_graph[i][j] = (1 - cosine_distance(
                np.array(list(weighted_frequencies.values()))[i],
                np.array(list(weighted_frequencies.values()))[j]))

# Use PageRank to calculate the importance of each sentence
sentence_ranks = nx.pagerank(nx.from_numpy_array(sentence_graph))

# Choose the top 3 ranked sentences as the summary
summary_length = 3
top_sentences = sorted(((sentence_ranks[i], s) for i, s in enumerate(filtered_sentences)), reverse=True)[:summary_length]
summary = ' '.join([s[1] for s in top_sentences])

# Write the summary to another file as a single paragraph
with open("C:/Users/Adavelli Rohan Reddy/Desktop/VIT/6th sem/RPA/Project/Email_Auto_modern/Email/output.txt", 'w') as file:
    file.write(summary.replace('\n', ' '))
