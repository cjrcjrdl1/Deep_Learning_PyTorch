import torch
print(torch.tensor([[1,2],[3,4]])) # 2차원 형태의 텐서 생성
print(torch.tensor([[1,2],[3,4]], device="cuda:0")) # GPU에 텐서 생성
print(torch.tensor([[1,2],[3,4]], dtype=torch.float64)) # dtype을 이용하여 텐서 생성

temp = torch.tensor([[1,2],[3,4]])
print(temp.numpy()) # 텐서를 ndarray로 변환

temp = torch.tensor([[1,2],[3,4]], device="cuda:0")
print(temp.to("cpu").numpy()) # GPU상의 텐서를 CPU의 텐서로 변환한 후 ndarray로 변환

# torch.FloatTensor: 32비트의 부동 소수점
# torch.DoubleTensor: 64비트의 부동 소수점
# torch.LongTensor: 64비트의 부호가 있는 정수
temp = torch.FloatTensor([1,2,3,4,5,6,7]) # 파이토치로 1차원 벡터 생성
print(temp[0],temp[1],temp[-1]) # 인덱스로 접근
print('-------------------------')
print(temp[2:5], temp[4:-1]) # 슬라이스로 접근

v = torch.tensor([1,2,3])
w = torch.tensor([3,4,6])
print(w-v)

temp = torch.tensor([[1,2],[3,4]])

print(temp.shape)
print("--------------------")
print(temp.view(4,1)) # 2x2행렬을 4x1로 변형
print('--------------------')
print(temp.view(-1)) # 2x2 행렬을 1차원 벡터로 변형
print('--------------------')
print(temp.view(1,-1)) # -1은 (1,?)와 같은 의미로 다른 차원으로부터 해당 값을 유추하겠다는 뜻
# temp의 원소 개수(2x2=4)를 유지한 채 (1, ?) 형태를 만족해야 하므로 (1,4)가 된다.
print('--------------------')
print(temp.view(-1,1)) # 앞에서와 마찬가지로 (?, 1)의 형태로 4개를 만들어내므로 (4,1)
