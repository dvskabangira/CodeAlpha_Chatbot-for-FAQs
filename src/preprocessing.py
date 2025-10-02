
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    filtered = [word for word in lemmatized if word not in stop_words]
    return filtered

