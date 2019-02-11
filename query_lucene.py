import os

import lucene
import numpy as np
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.store import SimpleFSDirectory
from sklearn.metrics.pairwise import cosine_similarity

from definitions import ROOT_DIR
from parser import Parser


class QueryLucene:
	"""Match documents from the corpus with queries using lucene"""

	def __init__(self, index_path='corpus/indexRI'):
		"""
		Lucene components initialization
		:param index_path: path of the index
		"""
		lucene.initVM()
		self.analyzer = StandardAnalyzer()
		self.index = SimpleFSDirectory(File(index_path).toPath())
		self.reader = DirectoryReader.open(self.index)
		self.searcher = IndexSearcher(self.reader)
		self.constrained_query = BooleanQuery.Builder()
		self.corpus = Parser.parsing_iot_corpus_pandas(os.path.join(ROOT_DIR, "corpus/iot-tweets-vector-v3.tsv"))

	def query_parser_filter(self, field_values, field_filter=['Vector']):
		"""
		Filtering queries according to field values
		:param field_values: values of the fields
		:param field_filter: fields to filter
		"""
		assert len(field_filter) == len(field_values), "Number of fields different from number of values"
		for i in range(len(field_filter)):
			query_parser = QueryParser(field_filter[i], self.analyzer)
			query = query_parser.parse(field_values[i])
			self.constrained_query.add(query, BooleanClause.Occur.FILTER)

	def query_parser_must(self, field_values, field_must=['Text']):
		"""
		The values that the fields must match
		:param field_values: values of the fields
		:param field_must: fields that must match
		"""
		assert len(field_must) == len(field_values), "Number of fields different from number of values"
		for i in range(len(field_must)):
			query_parser = QueryParser(field_must[i], self.analyzer)
			query = query_parser.parse(field_values[i])
			self.constrained_query.add(query, BooleanClause.Occur.MUST)

	def get_results(self, nb_results=1000):
		"""
		Get results that match with the query
		:param nb_results:
		:return:
		"""
		hits = self.searcher.search(self.constrained_query.build(), nb_results).scoreDocs
		self.constrained_query = BooleanQuery.Builder()
		return hits

	def get_docs_field(self, hits, field='Text'):
		"""
		Get a field from a query's results
		:param hits: the documents resulting from a query
		:param field: the field to get
		:return: the string value of a field from a query's results
		"""
		for i, hit in enumerate(hits):
			doc = self.searcher.doc(hit.doc)
			yield doc.getField(field).stringValue()

	def rerank_results(self, results, userVector):
		"""
		reranks the results of a query by using the similarity between the user thematic vector and the vector from the tweets
		:param results: the documents resulting from a query
		:param userVector: the thematic vector of a user
		:return: the reranked list of documents
		"""
		reranked = []
		for doc in results:
			doc_id = int(next(self.get_docs_field([doc], "TweetID")))
			doc_vector = self.corpus[self.corpus.TweetID == doc_id].Vector
			doc_vector = doc_vector.values[0] if len(doc_vector.values) > 0 else np.zeros(300)
			sim = cosine_similarity(userVector.reshape(1, -1), doc_vector.reshape(1, -1))
			reranked.append({'doc': doc, 'sim': sim[0][0]})

		reranked = sorted(reranked, key=lambda k: k['sim'], reverse=True)
		return [x['doc'] for x in reranked]

	def close_reader(self):
		self.reader.close()


if __name__ == '__main__':
	ql = QueryLucene()
	ql.query_parser_must(["First sign of twitter as transport for"])
	results = ql.get_results(10)
	docs = ql.rerank_results(results, np.zeros(300))
	texts = ql.get_docs_field(docs)
	for text in texts:
		print(text)