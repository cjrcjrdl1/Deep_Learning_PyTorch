import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv('./iris.data', names=names)

X = dataset.iloc[:, :-1].values #모든 행을 사용하지만 열은 뒤에서 하나를 뺀 값을 가져와서 X에 저장
y = dataset.iloc[:, 4].values # 모든 행을 사용하지만 열은 앞에서 다섯 번째 값만 가져와서 y에 저장

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)
# X,y를 사용하여 훈련과 테스트 데이터셋으로 분리하며, 테스트 데이터셋의 비율은 20%만 사용
from sklearn.preprocessing import StandardScaler
s = StandardScaler() #특성 스케일링, 평균이 0 표준편차가 1이 되도록 변환
X_train = s.fit_transform(X_train) #훈련 데이터를 스케일링 처리
X_test = s.fit_transform(X_test) #테스트 데이터를 스케일링 처리
#
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50) # K=50인 K-최근접 이웃 모델 생성
print(knn.fit(X_train, y_train)) #모델 훈련

from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
print("정확도: {}".format(accuracy_score(y_test,y_pred)))

k=10
acc_array = np.zeros(k)
for k in np.arange(1, k+1, 1):
    classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_array[k-1] = acc

max_acc = np.amax(acc_array)
acc_list = list(acc_array)
k = acc_list.index(max_acc)
print("정확도", max_acc, "으로 최적의 k는", k+1, "입니다.")