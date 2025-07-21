from ultralytics.nn.modules.block import AAM, Focus
import torch

model = Focus(c1=3, c2=, s=1)
x = torch.randn(size=(1, 3, 100, 100))
# print(model)
y = model(x)
print(y.shape)


