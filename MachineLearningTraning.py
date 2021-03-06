import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import numpy as np
from collections import Counter

style.use('ggplot')


# -----------------------------------------------------------------------------------------------------------------------------------------------
def handle_non_numerical_data(df):
	columns = df.columns.values
	for column in columns:
		text_digit_vals = {}

		def convert_to_int(val):
			return text_digit_vals[val]

		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x += 1
			df[column] = list(map(convert_to_int, df[column]))
	return df
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


class K_Means:
	def __init__(self, k=2, tol=0.001, max_iter=300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter



	def fit(self, data):
		self.centroids = {}

		for i in range(self.k):
			self.centroids[i] = data[i]

		for i in range(self.max_iter):
			self.classifications = {}

			for i in range(self.k):
				self.classifications[i] = []

			for featureset in data:
				distances = [np.linalg.norm(featureset - self.centroids[centroids]) for centroids in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)
			prev_centroids = dict(self.centroids)

			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)

			optimaized = True

			for c in self.centroids:
				original_centroids = prev_centroids[c]
				current_centroids = self.centroids[c]
				if np.sum((current_centroids - original_centroids) / original_centroids * 100.0) >= self.tol:
					optimaized = False

			if optimaized:
				break


	def predict(self, data):
		distances = [np.linalg.norm(data - self.centroids[centroids]) for centroids in self.centroids]
		classification = distances.index(min(distances))
		return classification
# -----------------------------------------------------------------------------------------------------------------------------------------------


class Mean_Shift:
	def __init__(self, radius=None, radius_norm_step=100):
		self.radius = radius
		self.radius_norm_step = radius_norm_step

	def fit(self, data):

		if self.radius is None:
			all_data_centroid = np.average(data, axis=0)
			all_data_norm = np.linalg.norm(all_data_centroid)
			self.radius = all_data_norm / self.radius_norm_step

		centroids = {}

		for i in range(len(data)):
			centroids[i] = data[i]

		weights = [i for i in range(self.radius_norm_step)][::-1]

		while True:
			new_centroids = []
			for i in centroids:
				in_bandwitdh = []
				centroid = centroids[i]
				for featureset in data:
					distances = np.linalg.norm(featureset - centroid)
					if distances == 0:
						distances = 0.00000000001
					weight_index = int(distances / self.radius)
					if weight_index > self.radius_norm_step - 1:
						weight_index = self.radius_norm_step - 1
					to_add = (weights[weight_index]**2) * [featureset]
					in_bandwitdh += to_add

				new_centroid = np.average(in_bandwitdh, axis=0)
				new_centroids.append(tuple(new_centroid))

			uniques = sorted(list(set(new_centroids)))

			to_pop = []

			for i in uniques:
				for ii in uniques:
					if i == ii:
						pass
					elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
						to_pop.append(ii)
						break
			for i in to_pop:
				try:
					uniques.remove(i)
				except:
					pass


			prev_centroids = dict(centroids)
			centroids = {}
			for i in range((len(uniques))):
				centroids[i] = np.array(uniques[i])
			optimaized = True

			for i in centroids:
				if not np.array_equal(centroids[i], prev_centroids[i]):
					optimaized = False
				if not optimaized:
					break
			if optimaized:
				break

		self.centroids = centroids


		self.classifications = {}

		for i in range(len(self.centroids)):
			self.classifications[i] = []
		for featureset in data:
			distances = [np.linalg.norm(featureset - self.centroids[centroid])for centroid in self.centroids]
			classification = distances.index(min(distances))
			self.classifications[classification].append(featureset)


	def predict(self, data):
		distances = [np.linalg.norm(featureset - self.centroids[centroid])for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

# -----------------------------------------------------------------------------------------------------------------------------------------------
