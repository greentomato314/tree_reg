import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class tree_reg:
    '''
    質的変数で場合分けをし、ぞれぞれで回帰モデルを作成。
    '''
    def __init__(self,df:pd.DataFrame,model,ycol,qualitative_col=[]):
        self.df = df.copy()
        self.cols = qualitative_col[:]

        self.dfls = self.data_split(self.df)
        self.modells = []
        for df1 in self.dfls:
            X = df1.drop(ycol,axis=1)
            X = X.drop(self.new_col,axis=1)
            Y = df1[ycol]
            self.modells.append(model().fit(X,Y))
        
    
    def data_split(self,df):
        self.new_col = '__'+''.join(self.cols)
        df1 = df.copy()
        df1[self.new_col] = ''
        for col in self.cols:
            df1[self.new_col] = df1[self.new_col] + df1[col].astype(str)
        df1 = df1.drop(self.cols,axis=1)
        vals = list(set(df1[self.new_col].to_list()))
        self.dic = dict()
        dfls = []
        cnt = 0
        for val in vals:
            self.dic[val] = cnt
            cnt += 1
            dfls.append(df1[df1[self.new_col]==val])
        return dfls

    def predict_all(self,df_p):
        df1 = df_p.copy()
        df1[self.new_col] = ''
        for col in self.cols:
            df1[self.new_col] = df1[self.new_col] + df1[col].astype(str)
        df1 = df1.drop(self.cols,axis=1)
        df1['predY'] = 0
        for key in self.dic.keys():
            df2 = df1[df1[self.new_col]==key]
            x = df2.drop([self.new_col,'predY'],axis=1)
            predY = self.modells[self.dic[key]].predict(x)
            df1.loc[df1[self.new_col]==key,'predY'] = predY
        return df1['predY'].values



if __name__=="__main__":
    df = pd.read_csv('dataL2.csv')
    print(df)
    TR = tree_reg(df,LinearRegression,ycol='y',qualitative_col=['class1','class2'])
    X = df.drop('y',axis=1)
    Y = df['y']
    pred = TR.predict_all(X)
    print(r2_score(Y,pred))
    plt.scatter(df['x'],df['y'])
    plt.scatter(df['x'],pred)
    plt.show()
    plt.scatter(Y,pred)
    plt.show()
    '''
    df = pd.read_csv('dataL.csv')
    TR = tree_reg(df,LinearRegression,ycol='y',qualitative_col=['class'])
    X = df.drop('y',axis=1)
    Y = df['y']
    pred = TR.predict_all(X)
    print(r2_score(Y,pred))
    plt.scatter(df['x'],df['y'])
    plt.scatter(df['x'],pred)
    plt.show()
    plt.scatter(Y,pred)
    plt.show()
    '''
    '''
    df = pd.read_csv('dataL.csv')
    df = df.drop('class',axis=1)
    print(df)
    TR = tree_reg(df,LinearRegression,ycol='y')
    X = df.drop('y',axis=1)
    Y = df['y']
    pred = TR.predict_all(X)
    print(r2_score(Y,pred))
    plt.scatter(df['x'],df['y'])
    plt.scatter(df['x'],pred)
    plt.show()
    plt.scatter(Y,pred)
    plt.show()
    '''