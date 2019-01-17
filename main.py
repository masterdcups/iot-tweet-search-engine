import gensim
import numpy as np
from parser import Parser
from user import User


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
	corpus = Parser.parsing_iot_corpus('corpus/fake-iot-corpus.tsv')

	for tweet in corpus['Text']:
		print(tweet)

	model = gensim.models.KeyedVectors.load_word2vec_format('corpus/GoogleNews-vectors-negative300.bin', binary=True)

	for tweet in corpus:
		print(list(tweet2vec(tweet, model)))

	exit()
	tweet_cliked_1 = tweet2vec(corpus[1], model)
	tweet_cliked_2 = tweet2vec(corpus[2], model)
	tweet_cliked_3 = tweet2vec(corpus[3], model)

	u1 = User()
	u1.update_profile(tweet_cliked_1)
	u1.save()

	u2 = User()
	u2.update_profile(tweet_cliked_2)
	u2.save()

	print(u2.vec)

	u3 = User()
	u3.update_profile(tweet_cliked_3)
	u3.save()

	u2 = User(2)
	u2.update_profile(tweet_cliked_3)
	u2.save()

	print(u2.vec)
