import lucene
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.index import DirectoryReader, MultiFields
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser

lucene.initVM()
analyzer = StandardAnalyzer()
index = SimpleFSDirectory(File("corpus/indexRI").toPath())
reader = DirectoryReader.open(index)
n_docs = reader.numDocs()
# 2. parse the query string
queryparser = QueryParser("Text", analyzer)
query = queryparser.parse("galette")
# 3. search the index
searcher = IndexSearcher(reader)
hits = searcher.search(query, n_docs).scoreDocs
# 4. display results
fields = MultiFields.getFields(reader)
iterator = fields.iterator()

while(iterator.hasNext()):
    field = iterator.next()
    terms = MultiFields.getTerms(reader, field)
    it = terms.iterator()
    term = it.next()
    while (term != None ):
        print(term.utf8ToString())
        term = it.next()
# 5. close resources
reader.close()
