from torch import optim as optim
optimizer=optim.SGD(model.parameters(), lr=.001 ,momentum=.9)