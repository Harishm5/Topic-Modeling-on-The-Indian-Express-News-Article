import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models

# 1. LOAD CSV FILE
csv_path = r"C:\Users\HARISH M\OneDrive\Desktop\News_Articles_Indian_Express.csv"
df = pd.read_csv(csv_path)

# 2. SELECT TEXT COLUMN
text_column = 'articles'  # Column containing article content
documents = df[text_column].dropna().astype(str).tolist()

# 3. TEXT PREPROCESSING
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())  # Tokenize text and convert to lowercase
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]  # Remove stop words and non-alphabetic tokens
    return tokens

processed_docs = [preprocess(doc) for doc in documents]

# 4. CREATE DICTIONARY AND CORPUS
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# 5. RUN LDA MODEL
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

# 6. SAVE VISUALIZATION TO HTML FILE
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_visualization.html')  # Save the visualization to an HTML file

print("Visualization saved as 'lda_visualization.html'. Open it in your browser.")
