tarin_dataset= barin-tumor-dataset ('/content/drive/MyDrive/brain_dataset','train')
test_dataset= barin-tumor-dataset ('/content/drive/MyDrive/brain_dataset','test')
from torch.utils.data import DataLoader
def new_concat(batch):
  return tuple(zip(*batch))
train_loader=DataLoader(tarin_dataset,batch_size=5
                        ,shuffle=True,collate_fn=new_concat)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True ,collate_fn=new_concat)
imgs,targets=next(iter(train_loader)
