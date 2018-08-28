import os 
import time
import math
import torch
import torch.nn as nn
from img_proc import *
from loss import cross_entropy2d,multiLoss_entropy2d
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from dataloader import MyTransform
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from resnet import ICNet,interp,Bottleneck,PSPDec

class CrossEntropyLoss2d(nn.Module):
	def __init__(self, weight=None):
		super().__init__()
		self.loss = nn.NLLLoss(weight)

	def forward(self, outputs, targets):
        	#torch version >0.2 F.log_softmax(input, dim=?) 
        	#dim (int): A dimension along which log_softmax will be computed.
		try:
			return self.loss(F.log_softmax(outputs,dim=1), targets) # if torch version >=0.3
		except TypeError as t:
			return self.loss(F.log_softmax(outputs), targets)       #else


def get_loader(data_txt):
    data_aug = Compose([RandomCrop(size=(360,640)), RandomHorizontallyFlip()])
    train_dataset=MyTransform(data_txt,augmentations=data_aug)
    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=8)
    return train_loader
def train(model,Para=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES=2
    num_epochs=62
    if Para:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    savedir="./log_save"
    weight = torch.ones(NUM_CLASSES)
    loader= get_loader("/home/user/ICNet-master/evaluation/list/train_640.txt")
    #if cuda:
       # criterion = CrossEntropyLoss2d(weight).cuda() 
    #else:
       # criterion = CrossEntropyLoss2d(weight)
    automated_log_path = savedir + "/automated_log.txt"
    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, 
                                momentum=0.9,
                                weight_decay=0.0005)           
    start_epoch = 1 
    loss_fn = cross_entropy2d
    for epoch in range(start_epoch, num_epochs):
        print("----- TRAINING - EPOCH", epoch, "-----")
        usedLr=0  
        epoch_loss = [] 
        time_train=[]
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])
        model.train()
        count = 1
        for step, (images, label1,label4,label24) in enumerate(loader):
            start_time = time.time()
            images = images.to(device)
            label1 = label1.to(device)
            label4 = label4.to(device)
            label24 = label24.to(device)            
            sub4,sub24,sub124 = model(images)#sub4,sub24,sub124
            
            loss1 = cross_entropy2d(sub4, label4)*0.16
            loss2 = cross_entropy2d(sub24, label24)*0.4
            loss3 = cross_entropy2d(sub124, label1)*1.0
            loss=loss1+loss2+loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()           
            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)
            if  step % 50 == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print('loss: {} (epoch: {}, step: {})'.format(average,epoch,step),"// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / 30))
        if  epoch % 10==0:
            torch.save(model.state_dict(), '{}_{}.pth'.format(os.path.join(savedir,"icnet"),str(epoch)))

        #save log
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f" % (epoch, average_epoch_loss_train))
    
    return(model) 
def main():
    cuda=True
    model=ICNet(Bottleneck,[3, 4, 6, 3])
    with open("./model.txt", "w") as myfile:  #record model 
        myfile.write(str(model))
    if cuda:
        model = model.cuda() 
    print("========== TRAINING ===========")
    model = train(model)
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    main()
