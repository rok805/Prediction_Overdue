#!/usr/bin/env python
# coding: utf-8

# ### 전처리

# In[8]:


# basic mudule
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math import isnan
import pandas as pd
import numpy as np
import os


# In[2]:


# 데이터 로드
cp = os.getcwd()
train = pd.read_csv(os.path.join(cp,'data/train.csv')).drop(['FLAG_MOBIL','index'], axis=1) # 변수 'FLAG_MOBIL','index' 제거
test = pd.read_csv(os.path.join(cp,'data/test.csv')).drop(['FLAG_MOBIL','index'], axis=1) # 변수 'FLAG_MOBIL','index' 제거


# In[3]:


def preprocessing_train(data):
    # 데이터 컬럼 타입 설정
    # object type
    data[['gender','car','reality','income_type','edu_type','family_type','house_type','work_phone','phone','email','occyp_type']] = data[['gender','car','reality','income_type','edu_type','family_type','house_type','work_phone','phone','email','occyp_type']].astype(object)

    # float type
    data[['child_num','income_total','DAYS_BIRTH','DAYS_EMPLOYED','family_size','begin_month']] = data[['child_num','income_total','DAYS_BIRTH','DAYS_EMPLOYED','family_size','begin_month']].astype(float)
    
    # DAYS_BIRTH 변수 나이로 변경.
    data['DAYS_BIRTH'] = [i/-365 for i in data['DAYS_BIRTH']]
    
    # DAYS_EMPLOYED 오류데이터
    tmp = data[data['income_type']!='Pensioner']
    m20 = tmp[tmp['DAYS_BIRTH'] < 30]['DAYS_EMPLOYED'].mean() # 20 대 평균 고용일수
    m30 = tmp[(tmp['DAYS_BIRTH'] >= 30) & (tmp['DAYS_BIRTH'] < 40)]['DAYS_EMPLOYED'].mean() # 30 대 평균 고용일수
    m40 = tmp[(tmp['DAYS_BIRTH'] >= 40) & (tmp['DAYS_BIRTH'] < 50)]['DAYS_EMPLOYED'].mean() # 40 대 평균 고용일수
    m50 = tmp[(tmp['DAYS_BIRTH'] >= 50) & (tmp['DAYS_BIRTH'] < 60)]['DAYS_EMPLOYED'].mean() # 50 대 평균 고용일수
    m60 = tmp[tmp['DAYS_BIRTH'] >= 60]['DAYS_EMPLOYED'].mean() # 60 대 평균 고용일수
    
    data.loc[(data.income_type == 'Pensioner')&(data.DAYS_BIRTH < 30), 'DAYS_EMPLOYED'] = m20
    data.loc[(data.income_type == 'Pensioner')&(data.DAYS_BIRTH >= 30) & (data.DAYS_BIRTH < 40), 'DAYS_EMPLOYED'] = m30
    data.loc[(data.income_type == 'Pensioner')&(data.DAYS_BIRTH >= 40) & (data.DAYS_BIRTH < 50), 'DAYS_EMPLOYED'] = m40
    data.loc[(data.income_type == 'Pensioner')&(data.DAYS_BIRTH >= 50) & (data.DAYS_BIRTH < 60), 'DAYS_EMPLOYED'] = m50
    data.loc[(data.income_type == 'Pensioner')&(data.DAYS_BIRTH >= 60), 'DAYS_EMPLOYED'] = m60
    
    # 직업이 없는 nan 분들 일단 'nojob' 으로 처리
    
    data['occyp_type'] = [i if isinstance(i,str) else 'nojob' for i in data['occyp_type']]
    
    data['begin_month'] = [i * -1 for i in data['begin_month']]
    
    # dummy variable 생성
    # X, y 분리s
    train_X = pd.get_dummies(data.drop(['credit'],axis=1))
    train_y = data['credit'].astype(int)

    return train_X, train_y
   


# In[4]:


train_X, train_y = preprocessing_train(train)


# In[190]:


def preprocessing_test(data):
    
    data[['gender','car','reality','income_type','edu_type','family_type','house_type','work_phone','phone','email','occyp_type']] = data[['gender','car','reality','income_type','edu_type','family_type','house_type','work_phone','phone','email','occyp_type']].astype(object)

    # float type
    data[['child_num','income_total','DAYS_BIRTH','DAYS_EMPLOYED','family_size','begin_month']] = data[['child_num','income_total','DAYS_BIRTH','DAYS_EMPLOYED','family_size','begin_month']].astype(float)
    
    # dummy variable 생성
    # X, y 분리
    X = pd.get_dummies(data)

    return X


# ### 데이터 분할

# In[179]:


def data_split(train_X, train_y):
    
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.3, shuffle=False, random_state=100)
    
    return np.array(train_X), np.array(valid_X), np.array(train_y), np.array(valid_y)


# In[ ]:


### 데이터 분할

def data_split(train_X, train_y):
    
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.3, shuffle=False, random_state=100)
    
    return np.array(train_X), np.array(valid_X), np.array(train_y), np.array(valid_y)

