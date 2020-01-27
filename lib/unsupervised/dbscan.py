# -*- coding: utf-8 -*-
"""
Created on 2020/1/27 6:36 下午

@Project -> File: clustering_algorithm -> dbscan.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: DBSCAN聚类
"""

from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd
import logging
import sys

logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt

sys.path.append('../..')

from lib import data


class DBSCANClustering(object):
	"""使用DBSCAN进行无监督聚类"""
	
	def __init__(self):
		pass
	
	def init_clstr(self, eps, min_samples = 5, **kwargs):
		"""
		初始化聚类器
		:param eps: float > 0, 同一簇中两个样本间的最大距离；
		:param min_samples: int > 0, default: 5, 一个簇中样本数目下限
		:param kwargs: 参见sklearn.cluster.DBSCAN其他关键字
		:return: clstr: sklearn.cluster.DBSCAN聚类器对象
		"""
		self.clstr = DBSCAN(eps = eps, min_samples = min_samples, **kwargs)
		return self.clstr
	
	def fit(self, data, show = False):
		y_pred = self.clstr.fit_predict(data)
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
	

def optim_params_search(data, eps_list, min_samples_list, eval_method):
	"""
	最优参数搜索
	:param eps_list: list of floats, 同簇样本最大间距list
	:param min_samples_list: list of ints, 同簇样本数下限
	:param eval_method: str, 聚类效果评估函数，参见KMeansClustering.cluster_effect_eval()
	:return: records: dict like {n_cluster_0: score_0, n_cluster_1: score_1, ...}
	"""
	records = {}
	dc = DBSCANClustering()
	for eps in eps_list:
		records[eps] = {}
		for min_samples in min_samples_list:
			_ = dc.init_clstr(eps = eps, min_samples = min_samples)
			y_pred = dc.fit(data)
			if len(set(y_pred)) == 1:
				score = np.nan
			else:
				score = dc.cluster_effect_eval(data, y_pred, method = eval_method)
			records[eps][min_samples] = score
	return records


if __name__ == '__main__':
	# %% 单次聚类和评测
	dc = DBSCANClustering()
	_ = dc.init_clstr(eps = 2.7, min_samples = 50)
	plt.figure('Clustering Effect')
	y_pred = dc.fit(data, show = True)
	plt.xticks(fontsize = 6)
	plt.yticks(fontsize = 6)
	score = dc.cluster_effect_eval(data, y_pred, method = 'calinski_harabasz')

	# %% 参数搜索
	import numpy as np
	eps_list = list(np.arange(0.5, 4.0, 0.1))
	min_samples_list = list(np.arange(5, 80, 10))
	methods = ['calinski_harabasz', 'silhouette', 'davies_bouldin']
	plt.figure('Clustering Evaluation')
	for method in methods:
		records = optim_params_search(data, eps_list, min_samples_list, eval_method = method)
		records = pd.DataFrame.from_dict(records)
		plt.subplot(3, 1, methods.index(method) + 1)
		mesh_x, mesh_y = np.meshgrid(eps_list, min_samples_list)
		plt.contourf(mesh_x, mesh_y, records)
		cb = plt.colorbar()
		cb.ax.tick_params(labelsize = 6)  # 设置色标刻度字体大小
		plt.title('{} score'.format(method), fontsize = 8, fontweight = 'bold')
		plt.xlabel('eps', fontsize = 8)
		plt.ylabel('min_samples', fontsize = 8)
		plt.xticks(fontsize = 6)
		plt.yticks(fontsize = 6)
		plt.tight_layout()
	plt.tight_layout()
	



