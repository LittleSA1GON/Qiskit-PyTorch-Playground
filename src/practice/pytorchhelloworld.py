import torch

torch.manual_seed(0)

# Data: y = 2x + 1
x = torch.linspace(-1, 1, 100).unsqueeze(1)   # (100, 1)
y = 2 * x + 1

model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for _ in range(200):
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()

print("weight:", model.weight.item())
print("bias:  ", model.bias.item())
print("loss:  ", loss.item()) 