import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import time
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from collections import OrderedDict
from resnet import ICNet,interp,Bottleneck,PSPDec

IS_MULTISCALE = True
N_CLASS = 2
is_local=True
inf_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.8]
data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])


class Inference(object):

    def __init__(self, model_path):
        self.seg_model =ICNet(Bottleneck,[3, 4, 6, 3])
        self.__load_weight(self.seg_model, model_path, is_local=is_local)
        self.seg_model = self.seg_model.cuda()

    def folder_inference(self, img_dir, is_multiscale=False):
        folders = sorted(os.listdir(img_dir))
        i=0
        for f in folders:
            i+=1
            read_path = os.path.join(img_dir, f)
            if not read_path.endswith(".png"):
                continue
            print(f)
            img = Image.open(read_path)
            img1=img.resize((640,360),Image.BILINEAR)
            if is_multiscale:
                pre = self.multiscale_inference(img1)
            else:
                pre = self.single_inference(img1)
            img=cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            mask = self.__pre_to_img(img,pre)
            if i>=30:
                break
            cv2.imwrite("./result/pred_"+str(f),mask)
            #cv2.imshow('DenseASPP', mask)
            #cv2.waitKey(0)
            

    def multiscale_inference(self, test_img):
        h, w = test_img.size
        pre = []
        for scale in inf_scales:
            img_scaled = test_img.resize((int(h * scale), int(w * scale)), Image.CUBIC)
            pre_scaled = self.single_inference(img_scaled, is_flip=False)
            pre.append(pre_scaled)

            img_scaled = img_scaled.transpose(Image.FLIP_LEFT_RIGHT)
            pre_scaled = self.single_inference(img_scaled, is_flip=True)
            pre.append(pre_scaled)

        pre_final = self.__fushion_avg(pre)

        return pre_final

    def single_inference(self, test_img, is_flip=False):
        torch.set_grad_enabled(False)
        image = Variable(data_transforms(test_img).unsqueeze(0).cuda())
        print("image_size",image.size())
        t1=time.time()
        sub4,sub24,pre = self.seg_model.forward(image)
        print("inference time:%d",time.time()-t1)
        if pre.size()[0] < 360:
            pre = F.upsample(pre, size=(360, 640), mode='bilinear')

        pre = F.log_softmax(pre, dim=1)
        pre = pre.data.cpu().numpy()
        if is_flip:
            tem = pre[0]
            tem = tem.transpose(1, 2, 0)
            tem = np.fliplr(tem)
            tem = tem.transpose(2, 0, 1)
            pre[0] = tem

        return pre

    @staticmethod
    def __fushion_avg(pre):
        pre_final = 0
        for pre_scaled in pre:
            pre_final = pre_final + pre_scaled
        pre_final = pre_final / len(pre)
        return pre_final

    @staticmethod
    def __load_weight(seg_model, model_path, is_local=True):
        print("loading pre-trained weight")
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)

        if is_local:
            seg_model.load_state_dict(weight)
        else:
            new_state_dict = OrderedDict()
            for k, v in weight.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            seg_model.load_state_dict(new_state_dict)

    @staticmethod
    def __pre_to_img(img,pre,color=[(34 ,139 ,34),(255,10,10)],alpha=0.5):
        result = np.array(pre.argmax(axis=1)[0])
        img=np.array(img)
        print(img.shape)
        for n, c in enumerate(color[0]):
            img[:, :, n] = np.where(
            result == 0,
            img[:, :, n] * (1 - alpha) + alpha * c,
            img[:, :, n]
        )
        return img
    
if __name__ =="__main__":
    model_path="./log_save/icnet_60.pth"
    infer=Inference(model_path)
    infer.folder_inference("/home1/image")
