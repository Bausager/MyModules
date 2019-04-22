import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import numpy as np
from collections import Counter


style.use('ggplot')


# -----------------------------------------------------------------------------------------------------------------------------------------------







# -----------------------------------------------------------------------------------------------------------------------------------------------



class Nearest_Neighbors:
	def __init__(self):
		pass

	def k_nearest_neighbors(self, data, predict, k=3):
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



# -----------------------------------------------------------------------------------------------------------------------------------------------



class Support_Vector_Machine:
	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1: 'r', -1: 'b'}
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1, 1, 1)

	# Training function
	def fit(self, data):
		self.data = data

		opt_dict = {}

		transform = [[1, 1],
					[-1, 1],
					[-1, -1],
					[1, -1]]

		all_data = []
		for yi in self.data:
			for featureset in self.data[yi]:
				for feature in featureset:
					all_data.append(feature)
		self.max_feature_value = max(all_data)
		self.min_feature_value = min(all_data)
		all_data = None

		# Support vectires = yi (xi.w + n) = 1
		step_sizes = [self.max_feature_value * 0.1,
					self.max_feature_value * 0.01,
					self.max_feature_value * 0.001,
					# point for expense:
					# self.max_feature_value * 0.0001,
					]

		# extremely expensive!
		b_range_multiple = 5
		# We don't need to take as small of steps with b as we do with w
		b_multiple = 5
		latest_optimum = self.max_feature_value * 10

		for step in step_sizes:
			w = np.array([latest_optimum, latest_optimum])

			# We can do this because of convex
			optimaized = False

			while not optimaized:
				# Start treading
				for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
										self.max_feature_value * b_range_multiple,
										step * b_multiple):
					for transformmation in transform:
						w_t = w * transformmation
						found_option = True
						# Weakest link in the SVM fundamentally.
						# SMO attempts to fix this a bit
						# yi (xi.w + b) >=1
						#
						# add a break later..
						for i in self.data:
							for xi in self.data[i]:
								yi = i
								if not yi * (np.dot(w_t, xi) + b) >= 1:
									found_option = False
						if found_option:
							opt_dict[np.linalg.norm(w_t)] = [w_t, b]
				# End treading
				if w[0] < 0:
					optimaized = True
					print('optimaized a step.')
				else:
					# w = [5, 5]
					# step = 1
					# w - [step, step] = [4, 4]
					# we can get away with w - step
					w = w - step
			norms = sorted([n for n in opt_dict])
			# opt_dict = ||w|| : [w, b]
			opt_choice = opt_dict[norms[0]]

			self.w = opt_choice[0]
			self.b = opt_choice[1]

			latest_optimum = opt_choice[0][0] + step * 2
		for i in self.data:
			for xi in self.data[i]:
				yi = i
				print(xi, ':', yi * (np.dot(self.w, xi) + self.b))


	def predict(self, features):
		classification = np.sign(np.dot(np.array(features), self.w) + self.b)
		if classification != 0 and self.visualization:
			self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
		return classification

		# For human interface, no bearing on the SVM
	def visualize(self, data_dict):
		[[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

		# hyperplane = x.w+b
		# v = x.w+b
		# psv = 1
		# nsv = -1
		# dec = 0


		def hyperplane(x, w, b, v):
			return (-w[0] * x - b + v) / w[1]

		datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
		hyp_x_min = datarange[0]
		hyp_x_max = datarange[1]

		# (w.x+b) = 1
		# positive support vector hyperplane
		psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
		psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
		self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

		# (w.x+b) = -1
		# negative support vector hyperplane
		nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
		nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
		self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

		# (w.x+b) = 0
		# positive support vector hyperplane
		db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
		db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
		self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

		plt.show()



# -----------------------------------------------------------------------------------------------------------------------------------------------
