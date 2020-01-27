# -*- coding: utf-8 -*-
"""
Created on 2020/1/27 2:01 下午

@Project -> File: clustering_algorithm -> kmeans.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: K-Means无监督聚类
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import copy
import sys

sys.path.append('../..')

from lib import data


class KMeansClustering(object):
	"""使用K-Means进行无监督聚类"""
	
	def setup(self):
		pass

	def init_clstr(self, n_clusters, init_method = 'k-means++', max_iter = 300, tol = 1e-4, random_state = 0, **kwargs):
		"""
		初始化聚类
		:param n_clusters: int > 0, 簇心数
		:param init_method: str in {'k-means++', 'random' or an ndarray}, default: 'k-means++', 簇心位置初始化方法
		:param max_iter: int > 0, default: 300, 最大迭代次数,
		:param tol: float, default: 1e-4, 收敛容差
		:param random_state: int, default: 0, 初始随机状态参数
		:return:
		"""
		self.clstr = KMeans(
			n_clusters = n_clusters, init = init_method, max_iter = max_iter, tol = tol, random_state = random_state, **kwargs)
		return self.clstr

	def fit(self, data, show = False):
		y_pred = self.clstr.fit_predict(data)
		# score = metrics.calinski_harabasz_score(data, y_pred)
		# score = metrics.silhouette_score(data, y_pred)
		# score = metrics.davies_bouldin_score(data, y_pred)
		if show:
			plt.scatter(data[:, 0], data[:, 1], c = y_pred, s = 6)
			plt.show()
		return y_pred
	
	@staticmethod
	def cluster_effect_eval(data, y_pred, method):
		"""
		:param method: str in {'calinski_harabasz', 'silhouette', 'davies_bouldin'}
		:return:
		"""
		if method == 'calinski_harabasz':
			score = metrics.calinski_harabasz_score(data, y_pred)
		elif method == 'silhouette':
			score = metrics.silhouette_score(data, y_pred)
		elif method == 'davies_bouldin':
			score = metrics.davies_bouldin_score(data, y_pred)
		else:
			raise ValueError('Unknown method {}.'.format(method))
		return score
	
	
def optim_params_search(data, n_clusters_list, eval_method):
	"""
	肘部搜索法
	:param n_clusters_list: list of ints, 聚类候选簇数list
	:param eval_method: str, 聚类效果评估函数，参见KMeansClustering.cluster_effect_eval()
	:return: records: dict like {n_cluster_0: score_0, n_cluster_1: score_1, ...}
	"""
	records = {}
	kmc = KMeansClustering()
	for n_clusters in n_clusters_list:
		_ = kmc.init_clstr(n_clusters = n_clusters)
		y_pred = kmc.fit(copy.deepcopy(data))
		score = kmc.cluster_effect_eval(data, y_pred, method = eval_method)
		records[n_clusters] = score
	return records


if __name__ == '__main__':
	# %% 单次聚类和评测
	kmc = KMeansClustering()
	_ = kmc.init_clstr(n_clusters = 7)
	plt.figure('Clustering Effect')
	y_pred = kmc.fit(data, show = True)
	plt.xticks(fontsize = 6)
	plt.yticks(fontsize = 6)
	score = kmc.cluster_effect_eval(data, y_pred, method = 'calinski_harabasz')
	
	# %% 参数搜索
	# 评测结果显示最优参数n_clusters
	import numpy as np
	n_clusters_list = list(np.arange(2, 16))
	methods = ['calinski_harabasz', 'silhouette', 'davies_bouldin']
	plt.figure('Clustering Evaluation')
	for method in methods:
		records = optim_params_search(data, n_clusters_list, eval_method = method)
		plt.subplot(3, 1, methods.index(method) + 1)
		plt.plot(list(records.keys()), list(records.values()))
		plt.xlabel('n_clusters', fontsize = 8)
		plt.ylabel('{} score'.format(method), fontsize = 8)
		plt.xticks(fontsize = 6)
		plt.yticks(fontsize = 6)
		plt.grid(True)
		plt.tight_layout()


