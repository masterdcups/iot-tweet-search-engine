import gensim
import numpy as np

from user import User


def read_corpus(fname):
	with open(fname, 'r') as f:
		for i, line in enumerate(f):
			yield gensim.utils.simple_preprocess(line)


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