import lucene
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader, MultiFields
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.store import SimpleFSDirectory

lucene.initVM()
analyzer = StandardAnalyzer()
index = SimpleFSDirectory(File("corpus/indexRI").toPath())
reader = DirectoryReader.open(index)
n_docs = reader.numDocs()
# 2. parse the query string
queryparser = QueryParser("Text", analyzer)
query = queryparser.parse("internet things")
# 3. search the index
searcher = IndexSearcher(reader)
# hits = searcher.search(query, 500).scoreDocs
# 4. display results
# for i, hit in enumerate(hits):
#     doc = searcher.doc(hit.doc)
#     print(doc.getField('Text').stringValue())

# filter with user vector (query his vector) IL FAUT LA COLONNE VECTOR DANS LE CORPUS
constrainedQuery = BooleanQuery.Builder()
constrainedQuery.add(query, BooleanClause.Occur.FILTER)

queryparser = QueryParser("Text", analyzer)
query = queryparser.parse("future")

constrainedQuery.add(query, BooleanClause.Occur.MUST)

hits = searcher.search(constrainedQuery.build(), 10).scoreDocs

for i, hit in enumerate(hits):
	doc = searcher.doc(hit.doc)
	print(doc.getField('Text').stringValue())

# 5. close resources
reader.close()
