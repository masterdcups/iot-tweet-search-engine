import os

import numpy as np
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.store import SimpleFSDirectory
from sklearn.metrics.pairwise import cosine_similarity

from definitions import ROOT_DIR
from models.tweet import Tweet
from parser import Parser
from profile_oneHotEncoder import ProfileOneHotEncoder


class QueryLucene:
	"""Match documents from the corpus with queries using lucene"""

	def __init__(self, index_path=os.path.join(ROOT_DIR, 'corpus/indexRI')):
		"""
		Lucene components initialization
		:param index_path: path of the index
		"""
		self.analyzer = StandardAnalyzer()
		self.index = SimpleFSDirectory(File(index_path).toPath())
		self.reader = DirectoryReader.open(self.index)
		self.searcher = IndexSearcher(self.reader)
		self.constrained_query = BooleanQuery.Builder()
		self.parser = Parser()

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

	def remove_duplicates(self, hits):
		"""
		remove duplicates (regarding the text field) from a scoreDocs object
		:param hits: the scoreDocs object resulting from a query
		:return: the scoreDocs object without duplicates
		"""
		seen = set()
		keep = []

		for i in range(len(hits)):
			if hits[i]["Text"] not in seen:
				seen.add(hits[i]["Text"])
				keep.append(hits[i])

		return keep

	def get_results(self, nb_results=1000):
		"""
		Get results that match with the query
		:param nb_results:
		:return:
		"""
		docs = self.searcher.search(self.constrained_query.build(), nb_results).scoreDocs
		self.constrained_query = BooleanQuery.Builder()

		hits = []
		for i in range(len(docs)):
			hits.append({})
			for field in self.reader.document(docs[i].doc).getFields():
				hits[i][field.name()] = field.stringValue()

		hits = self.remove_duplicates(hits)
		return hits

	def rerank_results(self, results, user_vector, user_gender, user_location, user_sentiment):
		"""
		reranks the results of a query by using the similarity between the user thematic vector and the vector from the tweets
		:param results: the documents resulting from a query
		:param user_vector: the thematic vector of a user
		:param user_gender: the gender of a user
		:param user_location: the location of a user
		:param user_sentiment: the sentiment of a user
		:return: the reranked list of documents
		"""
		reranked = []
		user_vec = ProfileOneHotEncoder.add_info_to_vec(user_vector, user_gender, user_location,
		                                                user_sentiment).reshape(1, -1)
		for i in range(len(results)):
			doc_infos = Tweet.load(int(results[i]['TweetID']))
			if doc_infos is None:
				reranked.append({'doc': results[i], 'sim': 0.})
			else:
				doc_vector = ProfileOneHotEncoder.add_info_to_vec(doc_infos.vector, doc_infos.gender, doc_infos.country,
				                                                  doc_infos.sentiment).reshape(1, -1)
				sim = cosine_similarity(user_vec, doc_vector)
				reranked.append({'doc': doc_infos, 'sim': sim[0][0]})
		reranked = sorted(reranked, key=lambda k: k['sim'], reverse=True)
		return [x['doc'] for x in reranked]

	def close_reader(self):
		self.reader.close()

	def link_tweets(self, results):
		return [Tweet.load(r['TweetID']) for r in results]


if __name__ == '__main__':
	ql = QueryLucene()
	ql.query_parser_must(["First sign of twitter as transport for"])
	results = ql.get_results()
	docs = ql.rerank_results(results, np.zeros(300), 'male', 'rw', 'neutral')[:10]
	for doc in docs:
		print(doc["Text"])
