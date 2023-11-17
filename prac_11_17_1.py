import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('sales data.csv')
print(data.head())

categorical_features = ['Channel', 'Region'] # 명목형 데이터
continuous_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_paper',
                       'Delicassen'] #연속형 데이터

for col in categorical_features:
    dummies = pd.get_dummies(data[col], prefix=col) # 명목형 데이터는 판다스의 get_dummies() 메서드를 사용하여 숫자(0과 1)로 변환
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)
print(data.head())

# 연속형 데이터의 모든 특성에 동일하게 중요성을 부여하기 위해 스케일링 적용
# 이는 데이터 범위가 다르기 때문에 범위에 따라 중요도가 달라질 수 있는 것
# 예를 들어 1000원과 1억이 있을 때 1000원을 무시하는 일을 방지하기 위함 사이킷런의 MinMaxScaler() 메서드 사용
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Optimal k')
plt.show()



