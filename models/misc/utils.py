import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def tokenize(text):
    word_list = word_tokenize(text.lower())
    only_words_list = [word for word in word_list if word.isalnum()]
    new_words = [WordNetLemmatizer().lemmatize(word) for word in only_words_list if
                 word not in stopwords.words('english')]

    return new_words
