"""
PCA:主成分分析过程：
    1.首先对样本矩阵进行去中心化：求出样本矩阵特征均值，然后减去特征均值。
        注意：样本矩阵行代表是样本个数，列代表的是样本的特征个数
    2.根据去中心化后的矩阵求出协方差矩阵
    3.求转换矩阵，求出特征值和相应的特征向量，从大网小取出K和特征值（k为降阶后的维度），并
        将特征值对应的特征向量组合成转换矩阵
    4.用原始矩阵乘以转换矩阵得出PCA后的输出矩阵
"""

import numpy as np

class PCA:
    def __init__(self, src_matrix, d):
        """
        初始化方法
        :param src_matrix: 需要进行PCA的原始矩阵
        :param d: 需要降维成d维的矩阵
        """
        self.src_matrix = src_matrix
        self.d = d

    def centralized(self, matrix):
        """
        中心化方法
        :param matrix: 需要进行处理的矩阵
        :return: 中心化后的矩阵
        """
        c_matrix = []
        r, c = np.shape(matrix)
        # 计算样本每一维特征的特征均值
        mean = [sum(matrix[:,i])/r for i in range(c)]
        c_matrix = matrix - mean
        print(c_matrix)
        return c_matrix  # 返回去中心化后的矩阵

    def cov(self, c_matrix):
        """

        :param c_matrix: 需要处理的矩阵
        :return: 协方差矩阵
        """
        r, c = np.shape(c_matrix)
        cov_m = np.dot(c_matrix.T, c_matrix)/(r - 1)
        print(cov_m)
        return cov_m # 返回协方差矩阵

    def trans(self, cov_m):
        """

        :param cov_m: 需要处理的矩阵
        :return: 转换矩阵
        """
        a, b = np.linalg.eig(cov_m)
        print("特征值为：\n", a)
        print("特征向量为：\n", b)
        ind = np.argsort(-1*a)
        trans_m =np.transpose([b[:,ind[i]] for i in range(self.d)])
        print(trans_m)
        return trans_m # 返回转换矩阵

    def output_m(self, trans_m):
        """

        :param trans_m: 转换矩阵
        :return: PCA处理后的最终结果
        """
        output_m = np.dot(self.src_matrix, trans_m)
        print(output_m)

if __name__ == '__main__':
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    pac = PCA(X, 2)
    c_matrix = pac.centralized(X)
    cov_m = pac.cov(c_matrix)
    trans_m = pac.trans(cov_m)
    pac.output_m(trans_m)





