import csv
import re

import gensim
import numpy as np
import preprocessor as p
from nltk.corpus import stopwords
from spellchecker import SpellChecker

from user import User


def replace_abbreviations(tokens):
    j = 0
    file_name = "corpus/slang.txt"
    with open(file_name, 'r') as myCSVfile:
        # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
        data_from_file = csv.reader(myCSVfile, delimiter="=")
        for token in tokens:
            # Removing Special Characters.
            _token = re.sub('[^a-zA-Z0-9-_.]', '', token)
            for row in data_from_file:
                # Check if selected word matches short forms[LHS] in text file.
                if token.upper() == row[0]:
                    # If match found replace it with its Abbreviation in text file.
                    tokens[j] = row[1]
            j = j + 1
        myCSVfile.close()
    return gensim.utils.simple_preprocess(' '.join(tokens))


def remove_stopwords_spelling_mistakes(spell, tokens):
    clean_tokens = []
    for token in tokens:
        # correction of spelling mistakes
        token = spell.correction(token)
        if token not in stopwords.words('english'):
            clean_tokens.append(token)
    return clean_tokens


def read_corpus(fname):
    # load spell checker
    spell = SpellChecker()
    # load lemmatizer
    # lmtzr = WordNetLemmatizer()

    with open(fname, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            tokens = []
            p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)
            tweet = p.clean(line)
            hashtags = list(part[1:] for part in tweet.split() if part.startswith('#'))
            tokens += gensim.utils.simple_preprocess(tweet) + gensim.utils.simple_preprocess(' '.join(hashtags))

            tokens = replace_abbreviations(tokens)
            tokens = remove_stopwords_spelling_mistakes(spell, tokens)
            # lemmatized_tokens = [lmtzr.lemmatize(word, 'v') for word in tokens]

            yield tokens
        f.close()


def tweet2vec(tweet, model):
    sentence_vector = []

    for word in tweet:
        try:
            sentence_vector.append(model.wv[word])

        except KeyError:
            pass

    # if a tweet word do not appear in the model we put a zeros vector
    if len(sentence_vector) == 0:
        sentence_vector.append(np.zeros_like(model.wv["tax"]))

    return np.mean(sentence_vector, axis=0)


if __name__ == '__main__':
    corpus = list(read_corpus('corpus/tweets.txt'))
    model = gensim.models.KeyedVectors.load_word2vec_format('corpus/GoogleNews-vectors-negative300.bin', binary=True)

    tweet_cliked_1 = tweet2vec(corpus[1], model)
    tweet_cliked_2 = tweet2vec(corpus[2], model)
    tweet_cliked_3 = tweet2vec(corpus[3], model)

    u1 = User()
    u1.update_profile(tweet_cliked_1, 'None', 'None', 'None')
    u1.save()

    u2 = User()
    u2.update_profile(tweet_cliked_2, 'None', 'None', 'None')
    u2.save()

    print(u2.vec)

    u3 = User()
    u3.update_profile(tweet_cliked_3, 'None', 'None', 'None')
    u3.save()

    u2 = User(2)
    u2.update_profile(tweet_cliked_3, 'None', 'None', 'None')
    u2.save()

    print(u2.vec)
