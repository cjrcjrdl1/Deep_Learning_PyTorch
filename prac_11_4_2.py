# import pandas as pd
# import torch
# data = pd.read_csv('../class2.csv')
#
# x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()
# y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()
# 각각의 CSV파일의 컬럼 값을 넘파이 배열로 받아 Tensor(dtype)으로 바꿔줌

# 한번에 메모리에 불러와서 훈련시ㅣ면 시간과 비용 측면에서 효율X
# 조금씩 나누어 불러서 사용하는 방식이 커스텀 데이터셋

import torch
import torchmetrics
metric = torchmetrics.Accuracy() #모델 평가(정확도) 초기화

n_batches = 10
for i in range(n_batches):
    preds = torch.randn(10,5).softmax(dim=1)
    target = torch.randint(5,(10,))

    acc = metric(preds, target)
    print(f"Accuracy on batch {i}: {acc}") # 현재 배치에서 모델 평가(정확도)

acc = metric.compute()
print(f"Accuracy on all data: {acc}") #모든 배치에서 모델 평가(정확도)
