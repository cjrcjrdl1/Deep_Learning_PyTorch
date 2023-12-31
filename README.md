# Deep_Learning_PyTorch
Deep_Learning_PyTorch

Chapter 1 (O)

Chapter 2 (O)

Chapter 3 (O)

비지도학습
```
K-평균 군집화
왜 사용? : 주어진 데이터에 대한 군집화
언제 사용하면 좋을까? : 주어진 데이터셋을 이용해서 몇개의 클러스터를 구성할지 사전에 알 수 있을 때 사용하면 유용
안 좋은 상황 : 데이터가 비선형일 때, 군집 크기가 다를 때, 군집마다 밀집도와 거리가 다를 때
```

```
밀집 기반 군집 분석
왜 사용? : 주어진 데이터에 대한 군집화
언제 사용하면 좋을까? : K-평균 군집화와 다르게 사전에 클러스터의 숫자를 알지 못할 때 사용하면 유용
                      또한, 주어진 데이터에 이상치가 많이 포함되었을 때 사용하면 좋음
노이즈는 주어진 데이터셋과 무관하거나 무작위성 데이터로 전치리 과정에서 제거해야 할 부분
이상치란 관측된 데이터 범위에서 많이 벗어난 아주 작은 값이나 아주 큰 값

```

```
주성분 분석(PCA)
왜 사용? : 주어진 데이터의 간소화
언제 사용하면 좋을까? : 현재 데이터의 특성(변수)이 너무 많은 경우에는 데이터를 하나의 플롯(plot)에 시각화해서 살펴보는 것이 어렵다.
                      이때 특성 p개를 두세 개 정도로 압축하여 데이터를 시각화하여 살펴보고 싶을 때 유용한 알고리즘
변수가 많은 고차원 데이터의 경우 중요하지 않은 변수로 처리해야 할 데이터 양 많아지고 성능 또한 나빠지는 경향이 있다.
이러한 문제를 해결하고자 고차원 데이터를 저차원으로 축소시켜 데이터가 가진 대표 특성만 추출한다면
성능은 좋아지고 작업도 좀 더 간편해짐
대표적인 알고리즘이 PCA -> 고차원 데이터를 저차원 데이터로 축소시키는 알고리즘
```

Chapter 4(O)
```
인공 신경망의 한계 -> AND, OR 는 선형 분류가 잘 되어 학습할 수 있지만,
XOR는 선형 분류를 할 수 없음 -> 해결방안은 중간에 은닉층을 둔 다층 퍼셉트론 사용
입력층과 출력층 사이에 여러 개의 신경망을 심층 신경망(DNN)이라고 하며 심층 신경망을 다른 말로 딥러닝이라고 함
```
```
딥러닝의 층은 입력층, 은닉층, 출력층으로 구성
가중치 : 노드와 노드 간 연결 강도
바이어스 : 가중치에 더해 주는 상수로, 하나의 뉴런에서 활성화 함수를 거쳐 최종적으로 출력되는 값 조절
가중합 : 가중치와 신호의 곱을 합한 것
활성화 함수 : 신호를 입력받아 이를 적절히 처리하여 출력해 주는 함수
손실 함수 : 가중치 학습을 위해 출력 함수의 결과와 실제 값 간의 오차를 측정하는 함수
```
```
활성화 함수
전달 함수에서 전달받은 값을 출력할 때 일정 기준에 따라 출력 값을 변화시키는 비선형 함수
시그모이드, 하이퍼볼릭 탄젠트, 렐루 함수 등이 있음

시그모이드 함수
선형 함수의 결과를 0~1 사이에서 비선형 형태로 변형해 준다.
주로 로지스틱 회귀와 같은 분류 문제를 확률적으로 표현하는 데 사용된다.
하지만 딥러닝 모델의 깊이가 깊어지면 기울기가 사라지는 '기울기 소멸 문제'가 발생하여 딥러닝에서 잘 사용X

하이퍼볼릭 탄젠트 함수
선형 함수의 결과를 -1~1 사이에서 비선형 형태로 변형해 준다.
시그모이드에서 결과 값 평균이 0이 아닌 양수로 편향된 문제를 해결하는 데 사용했지만,
기울기 소멸 문제가 여전히 발생

렐루 함수
입력(x)이 음수일 때는 0을 양수일 때는 x를 출력
경사 하강법에 영향을 주지 않아 학습 속도가 빠르고, 기울기 소멸 문제가 발생하지 않는 장점이 있음
렐루 함수는 일반적으로 은닉층에서 사용되며, 하이퍼볼릭 탄젠트 함수보다 6배 빠름
문제는 음수 값을 받으면 항상 0을 출력하기 때문에 학습 능력이 감소하는데,
이를 해결하려고 리키 렐루 함수 등을 사용

리키 렐루 함수
입력 값이 음수이면 0이 아닌 0.001처럼 매우 작은 수를 반환
이렇게 하면 입력 값이 수렴하는 구간이 제거되어 렐루 함수를 사용할 때 생기는 문제 해결

소프트맥스 함수
입력 값을 0~1 사이에 출력되도록 정규화하여 출력 값들의 총합이 항상 1이 되도록 한다.
소프트맥스 함수는 보통 딥러닝에서 출력 노드의 활성화 함수로 많이 사용된다.
exp(x)는 지수함수이다. n은 출력층의 뉴런 개수, yk는 그중 k번째 출력을 의미
```
```
손실 함수
경사 하강법은 학습률과 손실 함수의 순간 기울기를 이용하여 가중치를 업데이트하는 방법이다.
즉, 미분의 기울기를 이용해 오차를 비교하고 최소화하는 방향으로 이동시키는 방법
이때 오차를 구하는 방법이 손실 함수

즉, 손실 함수는 학습을 통해 얻은 데이터의 추정치가 실제 데이터와 얼마나 차이가 나는지 평가하는 지표
이 값이 클수록 많이 틀렸다는 의미고 0에 가까울수록 완벽하다는 것
대표적인 손실함수로는 평균 제곱 오차(MSE)와 크로스 엔트로피 오차(CEE)가 있다.

평균 제곱 오차
실제 값과 예측 값의 차이를 제곱하여 평균을 낸 것

크로스 엔트로피 오차
분류 문제에서 원-핫 인코딩 했을때만 사용할 수 있는 오차 계산법
일반적으로 분류 문제에서 데이터 출력을 0~1로 구분하기 위해 시그모이드 함수를 사용하는데,
시그모이드 함수에 포함된 자연 상수 e 때문에 평균 제곱 오차를 적용하면 매끄럽지 못한
그래프(울퉁불퉁한 그래프)가 출력된다. 따라서 크로스 엔트로피 손실 함수를 사용하는데,
이를 적용시 경사 하강법 과정에서 학습이 지역 최소점에서 멈출 수 있다. 이것을 방지하고자
자연 상수 e에 반대되는 자연 로그를 모델의 출력 값에 취한다.
```
```
딥러닝 학습 - 순전파, 역전파
순전파는 훈련 데이터가 들어올 때 발생
예측 값은 최종 층(출력층)에 도달하게 됨
손실함수로 네트워크 예측 값과 실제 값의 차이(손실, 오차)를 추정하는데 이때 손실 함수 비용은 0이 이상적
따라서 손실 함수 비용이 0에 가깝도록 하기 위해 무델이 훈련을 반복하면서 가중치를 조정
손실(오차)이 계산되면 그 정보는 역으로 전파(출력층 -> 은닉층 -> 입력층)되기 때문에 역전파라고 함
```
```
과적합
과적합은 훈련 데이터를 과하게 학습해서 발생
훈련 데이터에 대해 과하게 학습하여 실제 데이터에 대한 오차가 증가하는 현상

기울기 소멸 문제는 은닉층이 많은 신경망에서 주로 발생

기울기 소멸 발생시 오차가 줄어들면서 학습이 되지 않으므로, 성능이 나빠지는 문제 발생
이를 개선하고자 확률적 경사 하강법과 미니 배치 경사 하강법을 사용

배치 경사 하강법(BGD)은 전체 데이터셋에 대한 오류를 구한 뒤 기울기를 한 번만 계산하여 모델의 파라미터를 업데이트하는 방법
즉, 전체 훈련 데이터셋에 대해 가중치를 편미분하는 방법
배치 경사 하강법은 한 스텝에 모든 훈련 데이터셋을 사용하므로 학습이 오래 걸리는 단점이 있다.
오래걸리는 단점을 개선한 것이 확률적 경사 하강법

확률적 경사 하강법(SGD)은 임의로 선택한 데이터에 대해 기울기를 계산하는 방법으로 적은 데이터를 사용하므로 빠른 계산이 가능
때로는 배치 경사 하강법보다 정확도가 낮을 수 있지만 속도가 빠르다는 장점

미니 배치 경사 하강법은 전체 데이터셋을 미니 배치 여러개로 나누고, 미니 배치 한 개마다 기울기를 구한 후
그것을 평균 기울기를 이용해 모델을 업데이트해서 학습하는 방법
전체 데이터를 계산하는 것보다 빠르며, 확률적 경사 하강법보다 안정적이라는 장점이 있기에 많이 사용
안정적이며 속도 빠름

옵티마이저
확률적 경사 하강법의 파라미터 변경 폭이 불안정한 문제를 해결하기 위해 학습 속도와 운동량을 조정하는 옵티마이저를 적용해 볼 수 있다.
```
```
딥러닝 사용할 때 이점
특성 추출
데이터별로 어떤 특징을 가지고 있는지 찾아내고, 그것을 토대로 데이터를 벡터로 변환하는 작업을 특성 추출이라고 함
딥러닝에서는 특성 추출 과정을 알고리즘에 통합시킴. 데이터 특성을 잘 잡아내고자 은닉층을 깊게 쌓는 방식으로 파라미터를 늘린 모델 구조 덕분

빅데이터의 효율적 사용
딥러닝에서 특성추출을 알고리즘에 통합시키는 것이 가능한 이유는 빅데이터 때문
딥러닝 학습을 이용한 특성 추출은 데이터 사례가 많을수록 성능이 향상되기 때문

확보된 데이터가 적다면 딥러닝의 성능 향상을 기대하기 힘들기 때문에 머신러닝을 고려해 보아야 함
```
```
딥러닝 알고리즘은 심층 신경망을 사용한다는 공통점이 있다.
목적에 따라 합성곱 신경망(CNN), 순환 신경망(RNN), 제한된 볼츠만 머신(RBM), 심층 신뢰 신경망(DBN)으로 구성

심층 신경망(DNN)
입력층과 출력층 사이에 다수의 은닉층을 포함하는 인공 신경망
다수의 은닉층을 두기에 다양한 비선형적 관계를 학습할 수 있는 장점이 있지만,
학습응ㄹ 위한 연산량이 많고 기울기 소멸 문제 등이 발생할 수 있음
이를 해결하기 위해 드롭아웃, 렐루 함수, 배치 정규화 등을 적용해야 함

합성곱 신경망(CNN)
합성곱층과 풀링층을 포함하는 이미지 처리 성능이 좋은 인공 신경망 알고리즘
영상 및 사진이 포함된 이미지 데이터에서 객체를 탐색하거나 객체 위치를 찾아내는 데 유용한 신경망
대표적인 합성곱 신경망은 LeNet-5, AlexNet이 있다.
또한 층을 더 깊게 쌓은 신경망은 VGG, GoogLeNEt, ResNet 등이 있다.
특징
각 층의 입출력 형상을 유지
이미지 공간 정보를 유지하며 인접 이미지와 차이가 있는 특징을 효과적으로 인식
복수 필터로 이미지의 특징을 추출하고 학습
추출한 이미지의 특징을 모으고 강화하는 풀링층이 있다.
필터를 공유 파라미터로 사용하기 때문에 일반 인공 신경망과 비교하여 학습 파라미터가 매우 적다.

순환 신경망(RNN)
시계열 데이터(음악, 영상) 같은 시간 흐름에 따라 변화하는 데이터를 학습하기 위한 인공 신경망
순환은 자기 자신을 참조한다는 뜻으로 현재 결과가 이전 결과와 연관이 있다는 의미
특징
시간성을 가진 데이터가 많다.
시간성 정보를 이용하여 데이터의 특징을 잘 다룬다.
시간에 따라 내용이 변하므로 데이터는 동적이고 길이가 가변적이다.
매우 긴 데이터를 처리하는 연구가 활발히 진행되고 있다.
순환신경망은 기울기 소멸문제로 학습이 제대로 되지 않는 문제가 있다.
이를 해결하고자 메모리 개념을 도입한 LSTM이 순환 신경망에서 많이 사용
순환 신경망은 자연어 처리 분야와 궁합이 맞다.
대표적으로 언어 모델링, 텍스트 생성, 자동 번역(기계 번역), 음성 인식, 이미지 캡션 생성 등이 있다.

제한된 볼츠만 머신
가시층과 은닉층으로 구성된 모델
가시층은 은닉층과만 연결되는데(가시층과 가시층, 은닉층과 은닉층 사이에 연결이 없는) 이것이 제한된 볼츠만 머신이다.
특징
차원 감소, 분류, 선형 회귀 분석, 협업 필터링, 특성 값 학습, 주제 모델링에 사용
기울기 소멸 문제를 해결하기 위해 사전 학습 용도로 활용 가능
심층 신뢰 신경망(DBN)의 요소로 활용

심층 신뢰 신경망(DBN)
입력층과 은닉층으로 구성된 제한된 볼츠만 머신을 블록처럼 여러 층으로 쌓은 형태로 연결된 신경망
사전 훈련된 제한된 볼츠만 머신을 층층이 쌓아 올린 구조로, 레이블이 없는 데이터에 대한 비지도 학습이 가능
부분적인 이미지에서 전체를 연상하는 일반화와 추상화 과정을 구현할 때 사용하면 유용
```

Chapter 5 합성곱 신경망1 (O)
```
폴링층은 합성곱층과 유사하게 특성 맵의 차원을 다운 샘플링하여 연산량을 감소시키고, 주요한 특성 벡터를 추출하여 학습을 효과적으로 할 수 있다.
```

```
합성곱 신경망 : 이미지 전체를 한 번에 계산하는 것이 아닌 이미지의 국소적 부분을 계산함으로써
시간과 자원을 절약하여 이미지의 세밀한 부분까지 분석할 수 있는 신경망
```

Chapter 6 합성곱 신경망2 (O)
```
이미지는 3차원 구조를 가짐
이미지는 크기를 나타내는 너비 높이 뿐만 아니라 깊이도 가짐
색상이 보통 R/G/B 성분 세개를 갖기 때문에 시작이 3이지만 합성곱을
거치면서 특성 맵이 만들어지고 이것에 따라 중간 영상의 깊이가 달라짐

AlexNet은 합성곱층 총 다섯 개와 완전연결층 세개로 구성되어 있으며,
맨 마지막 완전연결층 카테고리 1000개를 분류하기 위해 소프트맥스 활성화 함수 사용

전체적으로 보면 GPU 두개를 기반으로 한 병렬 구조인 점을 제외하면
LeNet-5와 크게 다르지 않다.

GPU-1에서는 주로 컬러와 상관없는 정보를 추출하기 위한 커널이 학습되고,
GPU-2에서는 주로 컬러와 관련된 정보를 추출하기 위한 커널이 학습된다.

전반적인 코드는 LeNet과 크게 다르지 않기 때문에 주로 모델을 구성하는 네트워크가 어떻게 차이가 나는지 위주로 살펴보면 좋다.

ResNet
```

Chapter 7 RNN (ing......
