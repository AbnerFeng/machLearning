#-*-coding:utf-8-*-
from demo2.KNN import kNN

__author__ = 'Administrator'

#测试KNN
from numpy import *

group,labels = kNN.createDataSet()
#print group,labels
ans= kNN.classify([0,0],group,labels,3)
#print ans

reload(kNN)
datingDataMat,datingLabels= kNN.file2matrix('datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels)

#制作原始数据的散点图
import matplotlib.pyplot as plt
from pylab import *
#解决画图中文显示问题
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
xlabel(u"玩视频游戏所耗时间百分比", fontproperties=font)
ylabel(u"每周消费的冰激凌公升数", fontproperties=font)
# plt.show()

reload(kNN)

normMat,ranges,minVal = kNN.autoNorm(datingDataMat)
# print(normMat)
#
# kNN.datingClassTest()

kNN.classifyPerson()