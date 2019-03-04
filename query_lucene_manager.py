import os

from definitions import ROOT_DIR
from query_lucene import QueryLucene


class QueryLuceneManager:
	__instance = None

	@staticmethod
	def get_instance():
		""" Static access method. """
		if QueryLuceneManager.__instance is None:
			QueryLuceneManager.__instance = QueryLucene(corpus_path=os.path.join(ROOT_DIR,"corpus/iot-tweets-2009-2016-completv3.tsv"))
		return QueryLuceneManager.__instance

if __name__ == '__main__':
	ql = QueryLuceneManager.get_instance()
	ql.query_parser_must(["First sign of twitter as transport for"])
	results = ql.get_results(10)
	#docs = ql.rerank_results(results, np.zeros(300))[:10]
	for doc in results:
		print(doc["Text"])

	q1 = QueryLuceneManager.get_instance()
	ql.query_parser_must(["Not on Twitter? Sign up"])
	results = ql.get_results(10)
	# docs = ql.rerank_results(results, np.zeros(300))[:10]
	for doc in results:
		print(doc["Text"])