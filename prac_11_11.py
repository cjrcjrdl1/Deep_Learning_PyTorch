import pandas as pd
# 결정 트리
df = pd.read_csv('train.csv', index_col='PassengerId')
print(df.head()) # train.csv 데이터의 상위 행 다섯 개를 출력

df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df = df.dropna() #값이 없는 데이터 삭제
X = df.drop('Survived', axis=1)
y = df['Survived'] # Survived를 예측 레이블로 사용

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn import tree
model = tree.DecisionTreeClassifier()

model.fit(X_train, y_train) #모델 훈련

y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict)) #테스트 데이터에 대한 예측 결과를 보여줌

# 혼동행렬
from sklearn.metrics import confusion_matrix
print(pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival']
))

# True Positive: 모델(분류기)이 '1'이라고 예측했는데 실제 값도 '1'인 경우
# True Negative: 모델(분류기)이 '0'이라고 예측했는데 실제 값도 '0'인 경우
# False Positive: 모델(분류기)이 '1'이라고 예측했는데 실제 값도 '0'인 경우
# False Negative: 모델(분류기)이 '0'이라고 예측했는데 실제 값도 '1'인 경우

# 주어진 데이터를 사용하여 틜 형식으로 데이터를 이진분류(0 혹은 1)해 나가는 방법이 결정트리이며,
# 결정트리를 좀 더 확대한 것(결정 트리를 여러개 묶어 놓은 것)이 랜덤 포레스트이다.

# 로지스틱 회귀와 선형 회귀
# 회귀란 변수가 두개 주어졌을 때 한 변수에서 다른 변수를 예측하거나 두 변수의 관계를 규명하는데 사용
# 독립 변수(예측 변수) : 영향을 미칠 것으로 예상되는 변수
# 종속 변수(기준 변수) : 영향을 받을 것으로 예상되는 변수

# 예를 들어 몸무게(종속 변수)와 키(독립 변수)는 둘 간의 관계를 규명하는 용도로 사용

# 로지스틱 회귀
# 왜 사용할까? : 주어진 데이터에 대한 분류
# 언제 사용하면 좋을까? : 로지스틱 회귀 분석은 주어진 데이터에 대한 확신이 없거나
# 향후 추가적으로 훈련 데이터셋을 수집하여 모델을 훈련시킬 수 있는 환경에서 사용하면 유용

# 로지스틱 회귀는 분석하고자 하는 대상들이 두 집단 혹은 그 이상의 집단으로 나누어진 경우
# 개별 관측치들이 어떤 집단으로 분류될 수 있는지 분석하고 이를 예측하는 모형을 개발하는데 사용되는 통계 기법

# 구분            일반적인 회귀 분석      로지스틱 회귀 분석
# 종속변수          연속형 변수           이산형변수
# 모형 탐색 방법   최소제곱법              최대우도법
# 모형 검정       F-테스트, t-테스트      X^2 테스트

# %matplotlib inline
from sklearn.datasets import load_digits
digits = load_digits()
print("Image Data Shape", digits.data.shape)
# digits 데이터셋의 형태 (이미지가 1797개 있으며 8x8 이미지의1 64차원을 가짐)
print("Label Data Shape", digits.target.shape)
# 레이블(이미지 숫자 정보) 이미지 1797개가 있음

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)

plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(max_iter=5000) #로지스틱 회귀 모델의 인스턴스 생성
# max_iter 값을 크게 해주어야 오류 안남(서적에 표시 안되어있음)

logisticRegr.fit(x_train, y_train) #모델 훈련

print(logisticRegr.predict(x_test[0].reshape(1,-1)))
# 새로운 이미지(테스트 데이터)에 대한 예측 결과를 넘파이 배열로 출력
print(logisticRegr.predict(x_test[0:10])) #이미지 열 개에 대한 예측을 한 번에 배열로 출력

predictions = logisticRegr.predict(x_test) #전체 데이터셋에 대한 예측
score = logisticRegr.score(x_test, y_test) #스코어(socre) 메서드를 사용한 성능 측정
print(score)

import numpy as np
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions) #혼동 행렬
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt='.3f', linewidths=5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()

# 선형 회귀
# 왜 사용할까? : 주어진 데이터에 대한 분류
# 언제 사용하면 좋을까?
# 로지스틱 회귀는 주어진 데이터 독립 변수(x)와 종속 변수(y)가 선형 관계를 가질 때 사용하면 유용
# 또한 복잡한 연산 과정이 없기 때문에 컴퓨팅 성능이 낮은 환경(CPU/GPU 혹은 메모리 성능이 좋지 않을 때) 사용하면 좋음

# 하나의 x 값으로 y 값을 설명할 수 있다면 단순 선형 회귀
# x 값이 여러개라면 다중 선형 회귀

# 반면 로지스틱 회귀는 사건의 확률(0 또는 1)을 확인하는데 사용
# 예를 들어 구매할지 안할지 확인할 때 (1=예, 0=아니요)로 표현
# 선형 회귀는 직선을 출력(0~1의 범위를 초과할 수 있음)
# 로지스틱 회귀는 S-커브를 출력(0~1 범위 내에서만 존재)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('weather.csv')

dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print('평균제곱법:', metrics.mean_squared_error(y_test, y_pred))
print('루트 평균제곱법:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# 루트 평균 제곱법은 평균제곱법에 루트 씌운 것
