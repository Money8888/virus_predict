'''
数据清洗
'''
import pandas as pd
import os

# 数据清洗类
class DataClean():
    def __init__(self, name):
        # 加载数据资源进内存
        # name可选为confirmed, deaths, recovered
        fileStr = '/data/time_series_covid_19_' + name + '.csv'
        self.path = os.path.abspath(os.path.dirname(__file__)) + fileStr

    def read(self):
        # 读入数据成df,使用read_csv函数
        self.df = pd.read_csv(self.path, encoding='utf-8', header=0)

    # 查看三个Excel表结构一致，可以直接复用clean函数
    def clean(self):
        '''
        :return: 得到时间(x)-国家(y)的二维表
        '''
        # 删除不必要的列
        self.read()
        invalidCol = ['Province/State', 'Lat', 'Long']
        self.df = self.df.drop(invalidCol, axis=1)
        dayTime = list(self.df.columns)[1:]
        # 计算每个国家的每个地区的数据总和
        mergeDf = self.df.groupby('Country/Region')[dayTime].sum()
        # 转换数据框格式为 列名为各个国家地区，行名为每天的时间
        mergeDf =  pd.DataFrame(mergeDf.values.T, index=list(range(1, len(dayTime) + 1)), columns=mergeDf.index)
        # 时间序列数据正常，无须进行缺失值填补和异常值检测
        return mergeDf, dayTime


