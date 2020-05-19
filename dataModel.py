'''
数据建模类
'''
import datetime
import os

from math import ceil

import math
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Dense
from keras.losses import mean_squared_error
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dataClean import DataClean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

class Model_Function(object):
    def __init__(self):
        pass

    # 逻辑回归方程
    @staticmethod
    def logistics(t, P0, K, r):
        '''
        :param t: 时间，为1,2,3,4,...,等，分度值为天
        :param P0: 初始值
        :param K: 终态值
        :param r: 变化系数
        :return: 待拟合的参数
        '''
        T0 = 1
        fun = (K * np.exp(r * (t - T0)) * P0) / (K + (np.exp(r * (t - T0)) - 1) * P0)
        return fun

    @staticmethod
    def createLSTMDataSet(dataSet, windows=1):
        '''
        :param dataSet: 
        :param windows: 时间滑窗
        :return: 
        '''
        dataX, dataY = list(), list()
        for i in range(len(dataSet) - windows):
            dataX.append(dataSet[i:(i + windows)])
            dataY.append(dataSet[i + windows])
        return np.array(dataX), np.array(dataY)

    @staticmethod
    def timeMonthDay(dateList):
        timeList = []
        for time in dateList:
            splitTuple = time.split('/')
            if len(splitTuple[0]) <= 2:
                timeList.append(splitTuple[0] + '/' + splitTuple[1])
            if len(splitTuple[0]) > 2:
                timeList.append(splitTuple[0][2:] + '/' + splitTuple[1])
        return timeList

class DataModel():
    # name可选为confirmed, deaths, recovered
    # method取值为logistics或LSTM

    def __init__(self, name, method):
        # 根据name载入数据
        # 根据method选方法
        self.name = name
        dc = DataClean(self.name)
        self.df, self.dayTime = dc.clean()
        self.startDate = datetime.datetime.strptime(self.dayTime[0] + '20', '%m/%d/%Y').date()
        self.method = method

    def model(self, country):
        self.country = country
        time = list(range(1, len(self.df) + 1))
        # 解决画图时中文乱码问题
        from pylab import mpl
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        # 试测Arima模型
        # Arima模型忽略了确诊，死亡，治愈的关联关系，只是单纯依靠时间做的预测
        # 进行ADF检验
        '''
        pList = []
        for i in range(1, 10):
            series = self.df['China'].diff(2)
            series.fillna(0, inplace=True)
            p = adfuller(series)[1]
            pList.append({str(i): p})  # 计算p值
        '''
        # 发现1-10阶的差分的数据都通不过显著性检验，即p>0.05 故而舍弃Arima算法
        #
        # 试测logistics曲线模型
        # 设置参数范围
        if self.method == 'logistics':
            bounds = ([self.df[country][1], self.df[country][1], 0], [self.df[country][len(time)], self.df[country][len(time)], 1])
            # 开始最优化参数
            self.popt, self.pcov = curve_fit(Model_Function.logistics, time, self.df[country], bounds=bounds)
            x = self.dayTime
            y = self.df[country]
            y_pre = pd.Series([ceil(Model_Function.logistics(time[i], self.popt[0], self.popt[1], self.popt[2])) for i in range(len(time))])


        # 采用LSTM算法
        elif self.method == 'LSTM':
            # 设置随机种子
            np.random.seed(1)
            # 设置滑动窗口
            self.windows = 2
            # 标准化
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.dataSet = pd.DataFrame(self.scaler.fit_transform(pd.DataFrame(self.df[country])), index=time)
            #dataSet = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns, index=self.df.index)
            # 创建LSTM数据集
            trainX, trainY = Model_Function.createLSTMDataSet(self.dataSet[0], self.windows)
            # 对x进行扩维
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            # 搭建LSTM网络
            self.model = Sequential()
            self.model.add(LSTM(4, input_shape=(1, self.windows)))
            self.model.add(Dense(1))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            # 衰减学习率
            learn_rate = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='max')
            # with open(os.path.abspath(os.path.dirname(__file__)) + '/result/learn_rate.txt', 'a', encoding='UTF-8') as f:
            #     f.write(self.country + "的" + self.name + "LSTM模型的衰减学习率为" + str(learn_rate) + "\n")
            #     f.close()
            self.model.fit(trainX, trainY, epochs=100, batch_size=5, verbose=2, callbacks=[learn_rate])
            trainPredict = self.model.predict(trainX)
            # 反标准化
            trainPredict = self.scaler.inverse_transform(trainPredict)

            # 计算RMSE
            trainY = self.scaler.inverse_transform([trainY])
            trainScore = np.sqrt(sum((trainY[0, :] - trainPredict[:,0])**2)) / (len(trainY) * max(trainY[0, :]))
            with open(os.path.abspath(os.path.dirname(__file__)) + '/result/rmse.txt', 'a', encoding='UTF-8') as f:
                f.write(self.country + "的" + self.name + "预测与实际的RMSE为" + str(trainScore) + "\n")
                f.close()
            # print('Train Score: %.2f RMSE' % (trainScore))
            x = self.dayTime[self.windows:]
            y = self.df[country][self.windows:]
            y_pre = pd.Series(trainPredict[:,0]).astype(int)
        else:
            raise Exception('方法不存在!')

        x = Model_Function.timeMonthDay(x)
        plt.figure(figsize=[80, 60])
        plt.scatter(x, y, s=35, c='blue', marker='+', label=self.name + '人数')
        plt.plot(x, y_pre, 'r-s', marker='+', linewidth=1.5,  label="验证曲线")
        plt.tick_params(labelsize=8)
        plt.xlabel('日期', fontsize=2)
        plt.ylabel(self.name + '人数')
        x_major_locator = MultipleLocator(5)
        # 把x轴的刻度间隔设置为5，并存在变量里
        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        plt.legend(loc=0)
        plt.title(self.country + "的新冠病毒疫情的" + self.name + "发展及预测情况")
        #plt.show()

    # 预测指定天数的疫情情况，默认一周
    def predict(self, futureDay=7):
        preTime = np.linspace(len(self.df) + 1, len(self.df) + futureDay, num=futureDay)
        futureTime = [(self.startDate + datetime.timedelta(days=day)).strftime('%m/%d/%Y')[:-2] for day in preTime]
        if self.method == 'logistics':
            y = [ceil(Model_Function.logistics(preTime[i], self.popt[0], self.popt[1], self.popt[2])) for i in range(len(preTime))]
            y_pre = pd.Series(y)
        else:
            testX = [0] * (futureDay + self.windows)
            testX[0:self.windows] = self.dataSet[0][-self.windows:]
            testX = np.array(testX)
            testPre = [0] * futureDay
            for i in range(1, futureDay + 1):
                X = np.reshape(testX[i - 1 : i - 1 + self.windows], (1, 1, self.windows))
                tempY = self.model.predict(X)
                testX[self.windows + i - 1] = tempY
                testPre[i - 1] = tempY
            testPredict = np.array(testPre)
            testPredict = np.reshape(testPredict, (futureDay, 1))
            y_pre = pd.Series(self.scaler.inverse_transform(testPredict)[:,0]).astype(int)

        futureTime = Model_Function.timeMonthDay(futureTime)
        plt.scatter(futureTime, y_pre, s=35, c='green', label="预测人数")
        plt.ylabel(self.name + '人数')
        plt.legend(loc=0)

        # 记录结果数据
        with open(os.path.abspath(os.path.dirname(__file__)) + '/result/predict.txt', 'a', encoding='UTF-8') as f:
            for i in range(futureDay):
                currentTime = futureTime[i]
                f.write(self.country + "的第" + currentTime + "天的"  + self.name + "人数为" +str(y_pre[i]) + "\n")
            f.close()

    def view(self):

        plt.show()


if __name__ == '__main__':
    # name可选为confirmed, deaths, recovered
    dm = DataModel('deaths', 'LSTM')
    dm.model('Germany')
    dm.predict(45)
    dm.view()

    # dm = DataModel('confirmed', 'logistics')
    # dm.model('Whole World')
    # dm.predict(7)
    # dm.view()