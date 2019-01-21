from parser import Parser
from query_influencer_detection import QueryInfluencerDetection
from topics_classifier import TopicsClassifier

if __name__ == '__main__':
	parser = Parser()
	cleaned_tweet = parser.clean_tweet('smart watch')
	tweet_vec = parser.tweet2vec(cleaned_tweet)
	print(tweet_vec)

	clf = TopicsClassifier()
	topic_vec = clf.predict(tweet_vec.reshape(1, -1))

	qid = QueryInfluencerDetection()
	influencers = qid.get_influencers(topic_vec)
	print(influencers)
