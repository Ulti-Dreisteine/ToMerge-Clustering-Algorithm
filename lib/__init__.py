# -*- coding: utf-8 -*-
"""
Created on 2020/1/27 1:47 下午

@Project -> File: clustering_algorithm -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

from sklearn.datasets.samples_generator import make_blobs
import sys

sys.path.append('../')

from mod.config.config_loader import config

proj_dir = config.proj_dir
proj_cmap = config.proj_cmap

# 生成测试数据集
data, labels = make_blobs(
	n_samples = 2000,
	centers = 7,
	n_features = 5,
	cluster_std = 1,
	random_state = 0)
