''' para_op模块 原理：三分法找二次函数极点
输入：SVM训练数据集路径、测试数据集路径 C和gamma初始值
输出：C和gamma自动寻优的最终结果 '''

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

C_STRIDE = 0.2	# 指数步长
GAMMA_STRIDE = 0.2	# 指数步长

''' 测试当前C和gamma的分类准确率
输入：C和gamma
返回：测试集分类准确率 '''
def test(c, gamma, X, y):
	model = SVC(C=c, kernel='rbf', gamma=gamma)
	model.fit(X, y)
	scores = cross_val_score(model, X, y, cv=5).mean()
	return scores

''' 按照讨论结果进一步优化C '''
def tune_c(c_exp, gamma_exp, c_stride, X, y):
	# print('before c tuning, c_exp =', c_exp)
	# print('c_stride =', c_stride)
	c_exp_l = c_exp - c_stride
	c_exp_r = c_exp + c_stride
	c = np.exp2(c_exp)
	eta = test(np.exp2(c_exp), np.exp2(gamma_exp), X, y)
	eta_l = test(np.exp2(c_exp_l), np.exp2(gamma_exp), X, y)
	eta_r = test(np.exp2(c_exp_r), np.exp2(gamma_exp), X, y)

	if eta_l < eta and eta >= eta_r:
		c_stride = c_stride / 2
	elif eta_l < eta and eta < eta_r:
	    c_exp = c_exp_r
	elif eta_l == eta and eta > eta_r:
	    c_stride = c_stride / 2
	elif eta_l == eta and eta < eta_r:
	    c_exp = c_exp_r
	elif eta_l > eta and eta < eta_r:
	    if eta_l > eta_r:
	        c_exp = c_exp_l
	    else:
	        c_exp = c_exp_r
	elif eta_l > eta and eta >= eta_r:
	    c_exp = c_exp_l
	elif eta_l == eta and eta == eta_r:
		# print('shit!')
	    c_stride = c_stride / 2
	# print('after c tuning, c_exp =', c_exp)
	# print('c_stride =', c_stride)
	return c_exp, c_stride

''' 按照讨论结果进一步优化gamma '''
def tune_gamma(c_exp, gamma_exp, gamma_stride, X, y):
	# print('before gamma tuning, gamma_exp =', gamma_exp)
	# print('gamma_stride =', gamma_stride)
	gamma_exp_l = gamma_exp - gamma_stride
	gamma_exp_r = gamma_exp + gamma_stride
	eta = test(np.exp2(c_exp), np.exp2(gamma_exp), X, y)
	eta_l = test(np.exp2(c_exp), np.exp2(gamma_exp_l), X, y)
	eta_r = test(np.exp2(c_exp), np.exp2(gamma_exp_r), X, y)

	if eta_l < eta and eta >= eta_r:
		gamma_stride = gamma_stride / 2
	elif eta_l < eta and eta < eta_r:
	    gamma_exp = gamma_exp_r
	elif eta_l == eta and eta > eta_r:
	    gamma_stride = gamma_stride / 2
	elif eta_l == eta and eta < eta_r:
	    gamma_exp = gamma_exp_r
	elif eta_l > eta and eta < eta_r:
	    if eta_l > eta_r:
	        gamma_exp = gamma_exp_l
	    else:
	        gamma_exp = gamma_exp_r
	elif eta_l > eta and eta >= eta_r:
	    gamma_exp = gamma_exp_l
	elif eta_l == eta and eta == eta_r:
		# print('shit!')
	    gamma_stride = gamma_stride / 2
	# print('after gamma tuning, gamma_exp =', gamma_exp)
	# print('gamma_stride =', gamma_stride)
	return gamma_exp, gamma_stride

''' 主函数 '''
def optimize(X, y, init_log2C, init_log2gamma):
	log2_C = init_log2C
	log2_gamma = init_log2gamma
	c_stride = C_STRIDE
	gamma_stride = GAMMA_STRIDE

	log2_C, c_stride = tune_c(log2_C, log2_gamma, c_stride, X, y)
	log2_gamma, gamma_stride = tune_gamma(log2_C, log2_gamma, gamma_stride, X, y)
	# print(log2_C, log2_gamma)

	return (log2_C, log2_gamma)

	# c = np.exp2(c_exp)
	# gamma = np.exp2(gamma_exp)
	# print('C = {:.6f}, γ = {:.6f}时，测试集η = {:.6f}:'.format(c, gamma, test(c, gamma, X_train, y_train, X_test, y_test)))

