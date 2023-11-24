import numpy as np
print(np.random.randint(10))
print(np.random.randint(1,10))
print(np.random.rand(8))
print(np.random.rand(4,2))
print(np.random.randn(8))
print(np.random.randn(4,2))


examp = np.arange(0,100,3)
examp.resize(6,4)
print(examp)
print(examp[3])
print(examp[3,3])
print(examp[3][3])

examp = np.arange(0,500,3)
examp.resize(3,5,5)
print(examp)
print(examp[2][0][3])

abc= [1,2,3,4,5]
print(abc[0,:,:])