def format_matrice_likes():
	f1 = open('corpus/matrice_likes.csv', 'r')
	f2 = open('corpus/likes_matrix.tsv', 'w')

	f2.write(f1.readline())

	users = []

	for line in f1:
		parts = line[:-1].split(',')
		user_name = parts[0]
		if user_name not in users:
			users.append(user_name)
			tweets = ','.join(parts[1:])[2:-2].split(', ')

			f2.write(user_name + '\t' + '\t'.join(tweets) + '\n')

	f1.close()
	f2.close()


def format_follow_matrix():
	f1 = open('corpus/matrice_follower-following.tsv', 'r')
	f2 = open('corpus/followers_matrix.tsv', 'w')

	f1.readline()
	# f2.write(f1.readline())

	users = {}

	for line in f1:
		parts = line[1:-2].split('\t')

		user_name = parts[0]
		followers = [f[1:-1] for f in parts[1][1:-1].split(', ')]
		friends = [f[1:-1] for f in parts[2][1:-1].split(', ')]

		if user_name not in users:
			users[user_name] = followers

	for u in users:
		f2.write(u + '\t' + '\t'.join(users[u]) + '\n')

	f1.close()
	f2.close()


if __name__ == '__main__':
	format_follow_matrix()
