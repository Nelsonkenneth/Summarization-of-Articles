# Summarization-of-Articles
The "Article Summarization" project focuses on implementing automatic text summarization techniques using LexRank, NLTK, and TF-IDF algorithms. The project aims to condense lengthy articles or documents into concise summaries while retaining the most crucial information. Leveraging the power of Natural Language Processing (NLP) and statistical analysis, this project offers an efficient solution for digesting vast amounts of text quickly.

# Features:

1. LexRank Algorithm: Utilizes graph-based centrality scoring to identify important sentences based on their similarity to other sentences in the document.

2. NLTK (Natural Language Toolkit): Employs NLTK for text preprocessing tasks such as tokenization, stopword removal, and part-of-speech tagging, enhancing the quality of the summarization process.

3. TF-IDF (Term Frequency-Inverse Document Frequency): Computes the importance of words in a document relative to a corpus, enabling the identification of key terms and phrases for summarization.

4. Pandas: Data manipulation and analysis library

# Usage

Clone the repository to your local machine using 
```
 git clone https://github.com/Nelsonkenneth/Summarization-of-Articles.git
```
Install the necessary dependencies by running 
```
pip install -r requirements.txt
```
Ensure you have NLTK data downloaded by executing 
```
python -m nltk.downloader punkt stopwords.
```
Run the main script ``` summarization.py ``` and provide the path to the article or document you want to summarize.
View the generated summary output.

# Dependencies:

- Python 3.x
- NLTK
- NumPy
- Scikit-learn

# Contributing:
Contributions to enhance the project are welcome! Feel free to submit bug reports, feature requests, or pull requests via GitHub.

# License:
This project is licensed under the MIT License - ```LICENSE``` file for details.see the Dependencies:

# Acknowledgments:
Special thanks to the developers and contributors of NLTK, Scikit-learn, and other open-source libraries used in this project for their valuable contributions to the field of Natural Language Processing. And special thanks to my Professor for letting my do this amazing project.
