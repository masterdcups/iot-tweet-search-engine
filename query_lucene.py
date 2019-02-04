import lucene
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader, MultiFields
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.store import SimpleFSDirectory


class QueryLucene:
    """Match documents from the corpus with queries using lucene"""

    def __init__(self, index_path='corpus/indexRI'):
        """
        Lucene components initialization
        :param index_path: path of the index
        """
        self.lucene.initVM()
        self.analyzer = StandardAnalyzer()
        self.index = SimpleFSDirectory(File(index_path).toPath())
        self.reader = DirectoryReader.open(self.index)
        self.searcher = IndexSearcher(self.reader)
        self.constrained_query = BooleanQuery.Builder()

    def query_parser_filter(self, field_values, field_filter=['Vector']):
        """
        Filtering queries according to field values
        :param field_values: values of the fields
        :param field_filter: fields to filter
        """
        # TODO add an assert same length on field_filter and field_filter
        for i in range(len(field_filter)):
            query_parser = QueryParser(field_filter(i), self.analyzer)
            query = query_parser.parse(field_values(i))
            self.constrained_query.add(query, BooleanClause.Occur.FILTER)

    def query_parser_must(self, field_values, field_must=['Text']):
        """
        The values that the fields must match
        :param field_values: values of the fields
        :param field_must: fields that must match
        """
        # TODO add an assert same length on field_must and field_values
        for i in range(len(field_must)):
            query_parser = QueryParser(field_must(i), analyzer)
            query = query_parser.parse(field_values(i))
            self.constrainedQuery.add(query, BooleanClause.Occur.MUST)

    def get_results(self, nb_results=10):
        """
        Get results that match with the query
        :param nb_results:
        :return:
        """
        hits = self.searcher.search(self.constrainedQuery.build(), nb_results).scoreDocs
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

    def close_reader(self):
        self.reader.close()