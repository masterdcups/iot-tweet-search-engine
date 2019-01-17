from sklearn.decomposition import LatentDirichletAllocation

from parser import Parser


class LDAClassifier:

	def __init__(self):
		return

	def train(self):
		corpus = Parser.parsing_iot_corpus("path")
		X = corpus['Vector']
		y = corpus['TopicID']

		lda = LatentDirichletAllocation(n_components=5, random_state=0)
		lda.fit(X, y)
		lda.transform(X)


if __name__ == '__main__':
	lda_c = LDAClassifier()
	lda_c.train()
