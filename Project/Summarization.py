import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from LexRank import degree_centrality_scores
import pandas as pd
import time 

# Download the nltk punkt and stopwords to tokenize and ease the embeddings
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv(r"articles.csv")

start_time = time.time()

# Set the batch size for the process for it to go faster 
batch_size = 10

# The main function to compute the embeddings, find the similar and important sentences from the lexrank and print them
def summarize_text_batch(texts):
    summaries = []
    for text in texts:
        sentences = nltk.sent_tokenize(text)
        print("Num sentences:", len(sentences))
        
        # Use TfidfVectorizer to compute sentence embeddings
        vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Compute cosine similarity between sentences
        cos_scores = cosine_similarity(tfidf_matrix)
        
        # Compute the centrality for each sentence
        centrality_scores = degree_centrality_scores(cos_scores, threshold=None)
        
        # Get the indices of the most central sentences
        most_central_sentence_indices = np.argsort(-centrality_scores)[:10]
        
        # Print the 5 sentences with the highest scores
        summary = " ".join(sentences[idx].strip() for idx in most_central_sentence_indices)
        summaries.append(summary)
    return summaries

# Batch processing
num_batches = (len(df) + batch_size - 1) // batch_size
summaries = []
for i in range(num_batches):
    batch_texts = df["text"].iloc[i*batch_size : (i+1)*batch_size]
    summaries.extend(summarize_text_batch(batch_texts))

df["summary"] = summaries

print("Summarization completed in {:.2f} seconds".format(time.time() - start_time))

# Create a new dataframe containing the summaries and saves it 
result_df = pd.concat([df[["author", "title"]], df["summary"]], axis=1)
print(result_df.head())
result = result_df.to_csv("result.csv", index=False)
