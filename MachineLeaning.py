import warnings
import numpy as np
from collections import Counter


def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups!')
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			# euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
			distances.append([euclidean_distance, group])
	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k
	return vote_result, confidence
