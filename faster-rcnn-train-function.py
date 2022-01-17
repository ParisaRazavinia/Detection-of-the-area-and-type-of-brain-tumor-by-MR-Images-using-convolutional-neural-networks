import math 
def train_one_epoch(model,optimizer,train_loader):
  model.train()
  total_loss=0
  for images , targets in train_loader:
    images=[image.to(device) for image in images]
    targets=[{k: v.to(device ) for k , v in t.items()} for t in targets]
    loss_dict=model(images , targets )
    losses=sum(loss for loss in loss_dict.values())
    total_loss+=losses
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
  return total_loss/len(train_loader)
