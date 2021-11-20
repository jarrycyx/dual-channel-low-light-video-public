import torch.nn
import torch
import numpy as np

class ColorLoss(torch.nn.Module):
    def __init__(self, colorRatio=0.4, basicLossFunc=torch.nn.MSELoss()):
        super().__init__()
        self.colorRatio = colorRatio
        self.basicLossFunc = basicLossFunc
    
    def removeBrightness(self, img):
        R, G, B = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        L = (R.mul(R) + G.mul(G) + B.mul(B) + torch.ones(B.shape).to(B.device)*0.000001).sqrt()
        
        res = torch.zeros(img.shape).to(img.device)
        res[:, 0, :, :] = torch.div(img[:, 0, :, :], L)
        res[:, 1, :, :] = torch.div(img[:, 1, :, :], L)
        res[:, 2, :, :] = torch.div(img[:, 2, :, :], L)
        
        return res
        
    
    def colorLoss(self, inputImg, target):
        # inputImgColor = self.removeBrightness(inputImg)
        # targetColor = self.removeBrightness(target)
        cosEbLoss = torch.nn.CosineEmbeddingLoss(reduction = "mean")
        
        return cosEbLoss(inputImg.view(inputImg.shape[0],-1), target.view(inputImg.shape[0],-1), torch.ones(1).to(inputImg.device))
        # self.colorLossFunc(inputImgColor, targetColor)
                
         
    def forward(self, inputImg, target):
        basicLoss = self.basicLossFunc(inputImg, target)
        # punishment = self.calcPunishment(inputImg, target)
        colorLoss = self.colorLoss(inputImg, target)
        
        return basicLoss*(1-self.colorRatio) + colorLoss*self.colorRatio #+ punishment
    
    
if __name__ == "__main__":
    loss = ColorLoss()
    cosEbLoss = torch.nn.CosineEmbeddingLoss(reduction = "mean")
    
    x = torch.from_numpy(np.array([[[[1, 1], [1, 1]],
                                   [[1, 1], [1, 1]],
                                   [[1, 1], [1, 1]]]]).astype(np.float32))
    
    y = torch.from_numpy(np.array([[[[2, 1], [1, 1]],
                                   [[2, 1], [1, 1]],
                                   [[2, 1], [1, 1]]]]).astype(np.float32))
    
    print(loss(x, y))
    # print(x.shape)
    # print(loss(x, y))