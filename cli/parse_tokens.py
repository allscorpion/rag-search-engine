import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def get_stop_words():
    with open("data/stopwords.txt") as f:
        text = f.read()
        lines = text.splitlines()
        return lines


def convert_string_to_tokens(str):
    tokens = str.lower().translate(str.maketrans("", "", string.punctuation)).split()
    parsedTokens = list(filter(lambda token: token != "", tokens))
    return parsedTokens


def contains_token(tokenSet1, tokenSet2):
    for item in tokenSet1:
        for item2 in tokenSet2:
            if item in item2:
                return True
    return False


def filter_out_stop_words(stop_words, tokens):
    result = []
    for token in tokens:
        if token not in stop_words:
            result.append(token)

    return result


def stem_tokens(tokens):
    result = []
    for token in tokens:
        result.append(stemmer.stem(token))

    return result


def parse_tokens(str):
    stop_words = get_stop_words()
    tokens = convert_string_to_tokens(str)
    filtered_out_stop_words = filter_out_stop_words(stop_words, tokens)
    stemmed_tokens = stem_tokens(filtered_out_stop_words)

    return stemmed_tokens
