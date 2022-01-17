class barin-tumor-dataset (Dataset):
  def __init__(self,root,phase):
    self.root=root
    self.phase=phase
    self.imgs = os.listdir(os.path.join(root,"images"))
    self.targets = pd.read_csv(os.path.join(root,"data/{}_labels.csv".format(phase)))
    
  def __getitem__(self,idx):
    
    #images

    img_path=os.path.join(self.root,'images',self.imgs[idx])
    img=Image.open(img_path)
    img=F.to_tensor(img)

    #boxes

    box_list=self.targets[self.targets['filename']  == self.imgs[idx]]
    box_list = box_list[['xmin','ymin','xmax','ymax']].values
    boxes=torch.tensor(box_list, dtype = torch.float32)

    #labels
    labels=torch.ones((len(box_list,)),dtype=torch.int64)
    # make dictionary 

    target={}
    target['boxes']=boxes
    target['labels']=labels
    #
    return img , target 
  
  def __len__(self):

    return len(self.imgs)
