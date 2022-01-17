device = 'cuda' 
print(torch.cuda.get_device_properties(0))
model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.roi_heads.box_predictor=torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024,2)
model.to(device)