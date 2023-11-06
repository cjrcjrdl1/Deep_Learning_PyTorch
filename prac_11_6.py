import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv('./car_evaluation.csv')
dataset.head()

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
dataset.output.value_counts().plot(kind='pie', autopct='%0.05f%%',
                                   colors=['lightblue', 'lightgreen', 'orange', 'pink'],
                                   explode=(0.05,0.05,0.05,0.05))

# plt.show()
categorical_columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety'] #예제 데이터셋 칼럼들의 모곩

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category') # astype() 메서드를 이용하여 데이터를 범주형으로 변환

price = dataset['price'].cat.codes.values
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

categorical_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)
# print(categorical_data[:10]) #합친 넘파이 배열 중 열개의 행을 출력하여 보여줌

# 범주형 데이터를 텐서로 변환하기 위한 절차
# 범주형 데이터 -> dataset[category] -> 넘파이 배열 -> 텐서
# 즉, 파이토치로 모델을 학습시키기 위해서는 텐서 형태로 변환해야 하는데, 넘파이 배열을 통해 텐서를 생성할 수 있다.

# 범주형 데이터(단어)를 숫자(넘파이 배열)로 변환하기 위해 cat.codes를 사용
# cat.codes는 어떤 클래스가 어떤 숫자로 매핑되어 있는지 확인이 어려운 단점이 있으므로 주의
# np.stack은 두 개 이상의 넘파이 객체를 합칠 때 사용

# a = np.array([[1,2], [3,4]])
# b = np.array([[5,6], [7,8]])
# c = np.array([[5,6], [7,8], [9,10]])
# print(np.stack((a,b), axis=1))
# print(np.stack((a,c), axis=0)) # np.stack은 합치려는 두 넘파이 배열의1 차원이 같아야 함

categorical_data = torch.tensor(categorical_data, dtype=torch.int64)

outputs = pd.get_dummies(dataset.output)
outputs = outputs.values
outputs = torch.tensor(outputs).flatten()

# get_dummies는 가변수(dummy variable)로 만들어주는 함수로
# 가변수로 만든다는 의미는 문자를 숫자 (0,1)로 바꾸어 준다는 의미
# 예를 들어 성별, 몸무게, 국적 칼럼을 갖는 배열 생성

# data = {
#     'gender' : ['male', 'female', 'male'],
#     'weight' : [72,55,68],
#     'nation' : ['Japan', 'Korea', 'Australia']
# }
#
# df = pd.DataFrame(data)
# print(df)
#
# # 성별과 국적을 숫자로 변환하기 위해 get_dummies()를 적용
# print(pd.get_dummies(df))
#
# # ravel(), reshape(), flatten()은 텐서의 차원을 바꿀 때 사용
# a = np.array([[1,2], [3,4]])
# print(a.ravel())
# print(a.reshape(-1))
# print(a.flatten())

# 워드 임베딩은 유사한 단어끼리 유사하게 인코딩 되도록 표현하는 방법
# 높은 차원의 임베딩일수록 단어 간의 세부적인 관계를 잘 파악할 수 있다.
# 따라서 단일 숫자로 변환된 넘파이 배열을 N차원으로 변경하여 사용

# 임베딩 크기에 대한 정확한 규칙은 없지만, 칼럼을 고유 값 수를 2로 나누는 것을 많이 사용
# 예를 들어 price 칼럼은 4개의 고유 값을 갖기 때문에 임베딩 크기는 4/2 = 2이다.

# (모든 범주형 칼럼의 고유 값 수, 차원의 크기) 형태의 배열 출력 결과
categorical_columns_sizes = [len(dataset[column].cat.categories) for column in categorical_columns]
categorical_embedding_size = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_columns_sizes]

# 데이터셋을 훈련과 테스트 용도로 분리
total_records = 1728
test_records = int(total_records * .2) #전체 데이터 중 20%를 테스트 용도로 사용

categorical_train_data = categorical_data[:total_records-test_records]
categorical_test_data = categorical_data[total_records-test_records:total_records]
train_outputs = outputs[:total_records - test_records]
test_outputs = outputs[total_records - test_records : total_records]

class Model(nn.Module): # 클래스 형태로 구현되는 모델은 nn.Module을 상속받음
    def __init__(self, embedding_size, output_size, layers, p=0.4):
        # 모델에서 사용될 파라미터와 신경망을 초기화하기 위한 용도로 사용
        # embedding_size : 범주형 칼럼의 임베딩 크기
        # output_size : 출력층의 크기
        # layers : 모든 계층에 대한 목록
        # p : 드롭아웃(기본값은 0.5)

        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni,nf) for ni,
                                             nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols # 입력층의 크기를 찾기 위해 범주형 칼럼 개수를 input_size 변수에 저장

        for i in layers:
            all_layers.append(nn.Linear(input_size,i))
            # 선형 계층(linear layer)은 입력 데이터에 선형 변환을 진행한 결과
            # y = Wx + b
            all_layers.append(nn.ReLU(inplace=True))
            # 활성화 함수
            all_layers.append(nn.BatchNorm1d(i))
            # 배치 정규화 용도로 사용
            # 일반적으로 평균이 0, 분산이 1이 되도록 정규화
            all_layers.append(nn.Dropout(p))
            # 과적합 방지에 사용
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
            x = torch.cat(embeddings,1)
            x = self.embedding_dropout(x)
            x = self.layers(x)
            return x

model = Model(categorical_embedding_size, 4, [200,100,50], p=0.4)
print(model)

# 모델 훈련시키기 전 손실 함수와 옵티마이저에 대해 정의해야 함
# 크로스 엔트로피 손실함수 사용, 옵티마이저로는 아담을 사용
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if torch.cuda.is_available():
    device = torch.device('cuda') # GPU가 있다면 GPU를 사용
else:
    device = torch.device('cpu') # GPU가 없다면 CPU를 사용

