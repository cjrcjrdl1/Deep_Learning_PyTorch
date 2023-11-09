from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 환경변수를 사용하여 로깅을 제어(기본값은 0으로 모든 로그가 표시되며, INFO 로그를 필터링하려면 1. WARNING 로그를 필터링하려면 2 ERROR 로그를 추가로 필터링하려면 3으로 설정

iris = datasets.load_iris() #사이킷 런에서 제공하는 iris 데이터 호출
X_train, X_test, y_train, y_test = \
model_selection.train_test_split(iris.data,
                                 iris.target,
                                 test_size=0.6,
                                 random_state=42) #훈련 데이터와 테스트 데이터셋으로 분리

svm = svm.SVC(kernel='linear', C=1.0, gamma=0.5)
svm.fit(X_train, y_train) #훈련 데이터를 사용하여 SVM 분류기를 훈련
predictions = svm.predict(X_test) #훈련된 모델을 사용하여 테스트 데이터에서 예측
score = metrics.accuracy_score(y_test, predictions)
print('정확도: {0:f}'.format(score)) #테스트 데이터(예측) 정확도 측정

# SVM은 선형 분류와 비선형 분류를 지원. 비선형에 대한 커널은 선형으로 분류될 수 없는 데이터들 때문에 발생

# 비선형 문제를 해결하는 가장 기본적인 방법은 저차원 데이터를 고차원으로 보내는 것인데,
# 이것은 많은 수학적 계산이 필요하기 때문에 성능에 문제를 줄 수 있다.

# 이 문제를 해결하고자 도입한 것이 바로 커널 트릭이다.
# 선형 모델을 위한 커널에는 선형 커널이 있고
# 비선형 모델을 위한 커널에는 가우시안 RBF 커널과 다항식 커널이 있다.
# 가우시안 RBF 커널과 다항식 커널은 수학적 기교를 이용하는 것으로, 벡터를 계산한 후
# 고차원으로 보내는 방법으로 연산량을 줄임

# 선형 커널 : 선형으로 분류 가능한 데이터에 적용한다.
# K(a,b) = a^T * b (a,b : 입력 벡터)
# 선형 커널은 기본 커널 트릭으로 커널 트릭을 사용하지 않겠다는 의미와 일맥상통하다.

# 다항식 커널 : 실제로는 특성을 추가하지 않지만, 다항식 특성을 많이 추가한 것과 같은 결과를 얻을 수 있는 방법
# 즉, 실제로는 특성을 추가하지 않지만, 엄청난 수의 특성 조합이 생기는 것과 같은 효과를 얻기에
# 고차원으로 데이터 매핑이 가능
# K(a,b) = (감마a^T * b)^d (d는 차원, 여기서 감마, d는 하이퍼파라미터)

# 가우시안 RBF 커널 : 다항식 커널의 확장이라고 생각하자
# 입력 벡터를 차원이 무한한 고차원으로 매핑하는 것으로 모든 차수의 모든 다항식을 고려
# 즉 다항식 커널은 차수가 한계가 있는데, 가우시안 RBF는 차수에 제한 없이 무한한 확장 가능
# K(a,b) = exp(-감마 ||a-b||^2) (이때 감마는 하이퍼파라미터)

# C값은 오류를 어느정도 허용할지 지정하는 파라미터
# C 값이 클수록 하드마진 작을수록 소프트마진
# 감마는 결정경계를 얼마나 유연하게 가져갈지 지정
