ohist = []

ohist = [h.cpu().numpy() for h in hist]

plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy") 
plt.plot(range(1,num_epochs+1),ohist,label="resnet",color="maroon") 
plt.ylim((0,1.)) 
plt.xticks(np.arange(1, num_epochs+1, 1.0)) 
plt.legend() 
plt.show()