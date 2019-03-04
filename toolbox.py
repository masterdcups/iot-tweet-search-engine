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


if __name__ == '__main__':
	format_matrice_likes()
