import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__!='__main__':
    from .common import Conv,C3,C2f,SEAttention
else:
    from common import Conv,C3,C2f,SEAttention

__all__=["LPRNet"]



class LPRNet_c2f(nn.Module):
    def __init__(self,num_classes=66,dropout=0.0,se=False,*args,**kwargs):
        super().__init__()
        
        self.stem = nn.Sequential(
            Conv(3, 16, k=(3,5), s=(1,1),p=(1,0)),
            Conv(16, 32, k=(3,5), s=(1,1),p=(1,0)),
        )
        self.stage1 = nn.Sequential(
            Conv(32, 64, k=(3,5), s=(2,1),p=(1,0)),
            C2f(64, 64, n=3,g=1),
        )
        self.stage2 = nn.Sequential(
            Conv(64, 128, k=(3,3), s=2,p=(1,0)),
            C2f(128, 128, n=6,g=1),
        )
        self.stage3 = nn.Sequential(
            Conv(128, 256, k=3, s=2,p=(1,0)),
            C2f(256, 256, n=9,g=1),
        )
        self.neck = Conv(256, 512, k=(1, 3),p=0)
        self.dropout = nn.Dropout(p=dropout) if dropout>0.0 else nn.Identity()
        self.head = nn.Sequential(
            Conv(512, num_classes, k=1,p=0,act=False),
            nn.AvgPool2d((4, 1)),
        )
        
        self.se = None
        if se:
            self.se = SEAttention(512)
        
        self.export=False

    def forward(self, x):  # b, 3, 32, 96
        x = self.stem(x)     # b, 64, 32, 88
        x = self.stage1(x)   # b, 64, 16, 84
        x = self.stage2(x)   # b, 128, 8, 41
        x = self.stage3(x)   # b, 256, 4, 20
        x = self.neck(x)
        if self.se:
            x = self.se(x)
        x = self.dropout(x)
        x = self.head(x)
        
        if self.export:
            x = x.permute(0,2,3,1)
            x = F.softmax(x,dim=-1)
        return x                  # b, 68, 1, 18
    




    
def main():
    model = LPRNet_c2f()
    x = torch.randn((32, 3, 32, 96), dtype=torch.float32)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
