#-*- coding:utf-8 -*-
__author__ = 'Administrator'
#k邻近算法学习
#时间2014年5月7日20:00:05

#科学计算包NumPy
from numpy import *
#引入运算符模块
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#算法实现
#1）计算已知类别数据集中的点到当前点的距离
#2）按照距离递增次序排序
#3）选取与当前点距离最小的k个点
#4）确定前k个点所在类别的出现频率
#5）返回前k个点出现频率最高的类别作为当前点的预测类别
def classify(inX, dataSet, labels, k):
    #inX  用于分类的输入向量
    #dataSet 输入的训练样本集
    #labels  训练样本标签
    dataSetSize = dataSet.shape[0]
    #计算距离，tile为复制矩阵
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        vote_label = labels[sortedDistIndices[i]]
        classCount[vote_label] = classCount.get(vote_label,0)+1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#KNN   约会网站的匹配结果
#将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    fr = open (filename)
    arrayOLines = fr.readlines()
    #获得文本行数
    numberOLines = len(arrayOLines)
    returnMat = zeros((numberOLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    #参数为0，即从当前列中取得最小值
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal-minVal
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVal,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVal

#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.1
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVal = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVec = int (m*hoRatio)
    errCount = 0.0
    for i in range(numTestVec):
        classifierResult = classify(normMat[i,:],normMat[numTestVec:m,:],\
                                    datingLabels[numTestVec:m],3)
        print "the classifier came back with: %d,the real answer is: %d"\
                % (classifierResult,datingLabels[i])
        if ( classifierResult != datingLabels[i] ):
            errCount += 1.0
    print "the total error rate is :%f" % (errCount/float(numTestVec))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing videos games?"))
    ffMiles = float(raw_input("frequent flier miles earns per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVal = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr-minVal)/ranges,normMat,datingLabels,3)
    print "You will probably like the person:",resultList[classifierResult-1]


#读取数据#手写识别系统
def img2vector(filename):
    returnVec = zeros ((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0,32*i+j] = int(lineStr[j])
    return returnVec

from os import listdir

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult=classify(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with: %d,the real answer is :%d"\
                % (classifierResult,classNumStr)
        if (classifierResult != classNumStr):
            errorCount +=1.0
    print "\nthe total number of errors is :%d" %errorCount
    print "\nthe total error rate is :%f" % (errorCount/float(mTest))
