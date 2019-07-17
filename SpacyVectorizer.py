from sklearn.feature_extraction.text import CountVectorizer
import spacy
spacy.load('en')
class SpacyVectorizer(CountVectorizer):
    lemmatizer = spacy.lang.en.English()
    def __init__(self, stop_words, ngram_range, analyzer, max_features=None):
        super(SpacyVectorizer, self).__init__(stop_words = stop_words, ngram_range = ngram_range, \
                                              analyzer = analyzer, max_features = max_features, \
                                              tokenizer = SpacyVectorizer.l_tokenizer)
    @staticmethod
    def l_tokenizer(doc):
        tokens = SpacyVectorizer.lemmatizer(doc)
        return([token.lemma_ for token in tokens if not token.is_punct])