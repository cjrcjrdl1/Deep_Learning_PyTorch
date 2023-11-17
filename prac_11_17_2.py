import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
pd.set_option('display.max_columns', None)

X = pd.read_csv('credit card.csv')
X = X.drop('CUST_ID', axis=1) # 불러온 데이터에서 'CUST_ID' 열(칼럼)을 삭제
X.fillna(method='ffill', inplace=True)
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # 평균이 0, 표준편차가 1이 되도록 데이터 크기를 조정

X_normalized = normalize(X_scaled) #데이터가 가우스 분포를 따르도록 정규화
X_normalized = pd.DataFrame(X_normalized) #넘파이 배열을 데이터프레임으로 변환

pca = PCA(n_components=2) # 2차원으로 차원 축소 선언
X_principal = pca.fit_transform(X_normalized) # 차원 축소 적용
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
print(X_principal.head())

db_default = DBSCAN(eps=0.0375, min_samples=3).fit(X_principal) #모델 생성 및 훈련
labels = db_default.labels_ # 각 데이터 포인트에 할당된 모든 클러스터 레이블의 넘파이 배열을 labels에 저장

colours = {}
colours[0] = 'y'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k'

cvec = [colours[label] for label in labels] # 각 데이터 포인트에 대한 색상 벡터 생성
r = plt.scatter(X_principal['P1'], X_principal['P2'], color='y')
g = plt.scatter(X_principal['P1'], X_principal['P2'], color='g')
b = plt.scatter(X_principal['P1'], X_principal['P2'], color='b')
k = plt.scatter(X_principal['P1'], X_principal['P2'], color='k') #플롯의 범례 구성

plt.figure(figsize=(9,9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec) #정의된 색상 벡터에 따라 X축에 P1, Y축에 P2 플로팅

plt.legend((r,g,b,k), ('Label 0', 'Label 1', 'Label 2', 'Label -1')) # 범례 구축
plt.show()

# db = DBSCAN(eps=0.0375, min_samples=50).fit(X_principal)
db = DBSCAN(eps=0.0375, min_samples=100).fit(X_principal)
labels1 = db.labels_

colours1 = {}
colours1[0] = 'r'
colours1[1] = 'g'
colours1[2] = 'b'
colours1[3] = 'c'
colours1[4] = 'y'
colours1[5] = 'm'
colours1[-1] = 'k'

cvec = [colours1[label] for label in labels1]
colors1 = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

r = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[0])
g = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[1])
b = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[2])
c = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[3])
y = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[4])
m = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[5])
k = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[6])

plt.figure(figsize=(9,9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)
plt.legend((r,g,b,c,y,m,k),
           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'),
           scatterpoints = 1,
           loc='upper left',
           ncol=3,
           fontsize=8)
plt.show()