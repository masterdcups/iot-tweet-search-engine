from parser2 import Parser2
from query_influencer_detection import QueryInfluencerDetection
from topics_classifier import TopicsClassifier

if __name__ == '__main__':
	# Sample of getting influencers processus

	parser = Parser2()  # instance of parser

	# queries
	cleaned_tweet1 = parser.clean_tweet('smart watch #iot #watch')
	cleaned_tweet2 = parser.clean_tweet('connected light philips')
	cleaned_tweet3 = parser.clean_tweet('connected table and forks are comming')

	print(cleaned_tweet1)
	print(cleaned_tweet2)
	print(cleaned_tweet3)

	# convert query to w2v vectors
	tweet_vec1 = parser.tweet2vec(cleaned_tweet1)
	tweet_vec2 = parser.tweet2vec(cleaned_tweet2)
	tweet_vec3 = parser.tweet2vec(cleaned_tweet3)

	print(list(tweet_vec1))
	print(list(tweet_vec2))
	print(list(tweet_vec3))

	# convert w2v vectors to topics_vectors
	clf = TopicsClassifier()
	topic_vec1 = clf.predict(tweet_vec1.reshape(1, -1))
	topic_vec2 = clf.predict(tweet_vec2.reshape(1, -1))
	topic_vec3 = clf.predict(tweet_vec3.reshape(1, -1))

	# get the influencers from topics_vectors
	qid = QueryInfluencerDetection()
	influencers_query1 = qid.get_influencers(topic_vec1)
	influencers_query2 = qid.get_influencers(topic_vec2)
	influencers_query3 = qid.get_influencers(topic_vec3)

	print(influencers_query1)
	print(influencers_query2)
	print(influencers_query3)
