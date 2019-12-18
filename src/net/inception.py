import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

    
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = BasicConv2d(in_planes, n1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv branch
        self.b2_1x1_a = BasicConv2d(in_planes, n3x3red, 
                                    kernel_size=1)
        self.b2_3x3_b = BasicConv2d(n3x3red, n3x3, 
                                    kernel_size=3, padding=1)

        # 1x1 conv -> 3x3 conv -> 3x3 conv branch
        self.b3_1x1_a = BasicConv2d(in_planes, n5x5red, 
                                    kernel_size=1)
        self.b3_3x3_b = BasicConv2d(n5x5red, n5x5, 
                                    kernel_size=3, padding=1)
        self.b3_3x3_c = BasicConv2d(n5x5, n5x5, 
                                    kernel_size=3, padding=1)

        # 3x3 pool -> 1x1 conv branch
        self.b4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.b4_1x1 = BasicConv2d(in_planes, pool_planes, 
                                  kernel_size=1)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2_3x3_b(self.b2_1x1_a(x))
        y3 = self.b3_3x3_c(self.b3_3x3_b(self.b3_1x1_a(x)))
        y4 = self.b4_1x1(self.b4_pool(x))
        return torch.cat([y1, y2, y3, y4], 1)
    
class GoogLeNet(nn.Module):
    def __init__(self, classes):
        super(GoogLeNet, self).__init__()
        self.pre_layers = BasicConv2d(3, 192, kernel_size=3, padding=1)

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1024, classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.sigmoid(out)
    
class PSGoogLeNet(nn.Module):
    def __init__(self, classes):
        super(PSGoogLeNet, self).__init__()
        self.pre_layers = BasicConv2d(3, 192, kernel_size=3, padding=1)

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv_whole = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.conv_hs = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.conv_ub = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.conv_lb = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.conv_sh = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.conv_at = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.fc_whole = nn.Linear(1024, classes[0])
        self.fc_hs = nn.Linear(1024, classes[1])
        self.fc_ub = nn.Linear(1024, classes[2])
        self.fc_lb = nn.Linear(1024, classes[3])
        self.fc_sh = nn.Linear(1024, classes[4])
        self.fc_at = nn.Linear(1024, classes[5])

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        
        attention_whole = self.conv_whole(out)
        attention_whole = F.sigmoid(attention_whole)
        out_whole = out * attention_whole
        out_whole = self.avgpool(out_whole)
        out_whole = out_whole.view(out_whole.size(0), -1)
        out_whole = self.fc_whole(out_whole)
        
        attention_hs = self.conv_hs(out)
        attention_hs = F.sigmoid(attention_hs)
        out_hs = out * attention_hs
        out_hs = self.avgpool(out_hs)
        out_hs = out_hs.view(out_hs.size(0), -1)
        out_hs = self.fc_hs(out_hs)
        
        attention_ub = self.conv_ub(out)
        attention_ub = F.sigmoid(attention_ub)
        out_ub = out * attention_ub
        out_ub = self.avgpool(out_ub)
        out_ub = out_ub.view(out_ub.size(0), -1)
        out_ub = self.fc_ub(out_ub)
        
        attention_lb = self.conv_lb(out)
        attention_lb = F.sigmoid(attention_lb)
        out_lb = out * attention_lb
        out_lb = self.avgpool(out_lb)
        out_lb = out_lb.view(out_lb.size(0), -1)
        out_lb = self.fc_lb(out_lb)
        
        attention_sh = self.conv_sh(out)
        attention_sh = F.sigmoid(attention_sh)
        out_sh = out * attention_sh
        out_sh = self.avgpool(out_sh)
        out_sh = out_sh.view(out_sh.size(0), -1)
        out_sh = self.fc_sh(out_sh)
        
        attention_at = self.conv_at(out)
        attention_at = F.sigmoid(attention_at)
        out_at = out * attention_at
        out_at = self.avgpool(out_at)
        out_at = out_at.view(out_at.size(0), -1)
        out_at = self.fc_at(out_at)
        
        return F.sigmoid(torch.cat([out_whole, out_hs, out_ub, out_lb, out_sh, out_at], 1))
    
class PCGoogLeNet(nn.Module):
    def __init__(self, classes):
        super(PCGoogLeNet, self).__init__()
        self.pre_layers = BasicConv2d(3, 192, kernel_size=3, padding=1)

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_whole = nn.Linear(1024, 1024)
        self.linear_ub = nn.Linear(1024, 1024)
        self.linear_lb = nn.Linear(1024, 1024)
        self.linear_hs = nn.Linear(1024, 1024)
        self.linear_sh = nn.Linear(1024, 1024)
        self.linear_at = nn.Linear(1024, 1024)
        self.fc_whole = nn.Linear(1024, classes[0])
        self.fc_hs = nn.Linear(1024, classes[1])
        self.fc_ub = nn.Linear(1024, classes[2])
        self.fc_lb = nn.Linear(1024, classes[3])
        self.fc_sh = nn.Linear(1024, classes[4])
        self.fc_at = nn.Linear(1024, classes[5])

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        
        attention_whole = self.avgpool(out)
        attention_whole = attention_whole.view(attention_whole.size(0), -1)
        attention_whole = self.linear_whole(attention_whole)
        attention_whole = F.sigmoid(attention_whole)
        attention_whole = attention_whole.view((-1, attention_whole.size(1), 1, 1))
        #print("*********:  ", attention_whole.size())
        out_whole = out * attention_whole
        out_whole = self.avgpool(out_whole)
        out_whole = out_whole.view(out_whole.size(0), -1)
        out_whole = self.fc_whole(out_whole)
        #print("*********:  ", out_whole.size())
        
        attention_hs = self.avgpool(out)
        attention_hs = attention_hs.view(attention_hs.size(0), -1)
        attention_hs = self.linear_hs(attention_hs)
        attention_hs = F.sigmoid(attention_hs)
        attention_hs = attention_hs.view((-1, attention_hs.size(1), 1, 1))
        out_hs = out * attention_hs
        out_hs = self.avgpool(out_hs)
        out_hs = out_hs.view(out_hs.size(0), -1)
        out_hs = self.fc_hs(out_hs)
        
        attention_ub = self.avgpool(out)
        attention_ub = attention_ub.view(attention_ub.size(0), -1)
        attention_ub = self.linear_ub(attention_ub)
        attention_ub = F.sigmoid(attention_ub)
        attention_ub = attention_ub.view((-1, attention_ub.size(1), 1, 1))
        out_ub = out * attention_ub
        out_ub = self.avgpool(out_ub)
        out_ub = out_ub.view(out_ub.size(0), -1)
        out_ub = self.fc_ub(out_ub)
        
        attention_lb = self.avgpool(out)
        attention_lb = attention_lb.view(attention_lb.size(0), -1)
        attention_lb = self.linear_lb(attention_lb)
        attention_lb = F.sigmoid(attention_lb)
        attention_lb = attention_lb.view((-1, attention_lb.size(1), 1, 1))
        out_lb = out * attention_lb
        out_lb = self.avgpool(out_lb)
        out_lb = out_lb.view(out_lb.size(0), -1)
        out_lb = self.fc_lb(out_lb)
        
        attention_sh = self.avgpool(out)
        attention_sh = attention_sh.view(attention_sh.size(0), -1)
        attention_sh = self.linear_sh(attention_sh)
        attention_sh = F.sigmoid(attention_sh)
        attention_sh = attention_sh.view((-1, attention_sh.size(1), 1, 1))
        out_sh = out * attention_sh
        out_sh = self.avgpool(out_sh)
        out_sh = out_sh.view(out_sh.size(0), -1)
        out_sh = self.fc_sh(out_sh)
        
        attention_at = self.avgpool(out)
        attention_at = attention_at.view(attention_at.size(0), -1)
        attention_at = self.linear_at(attention_at)
        attention_at = F.sigmoid(attention_at)
        attention_at = attention_at.view((-1, attention_at.size(1), 1, 1))
        out_at = out * attention_at
        out_at = self.avgpool(out_at)
        out_at = out_at.view(out_at.size(0), -1)
        out_at = self.fc_at(out_at)
        
        return F.sigmoid(torch.cat([out_whole, out_hs, out_ub, out_lb, out_sh, out_at], 1))
    
class PGoogLeNet(nn.Module):
    def __init__(self, classes):
        super(PGoogLeNet, self).__init__()
        self.pre_layers = BasicConv2d(3, 192, kernel_size=3, padding=1)

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv_whole = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.conv_hs = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.conv_ub = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.conv_lb = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.conv_sh = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.conv_at = nn.Conv2d(1024, 1, kernel_size=1, padding=1)
        self.linear_whole = nn.Linear(1024, 1024)
        self.linear_ub = nn.Linear(1024, 1024)
        self.linear_lb = nn.Linear(1024, 1024)
        self.linear_hs = nn.Linear(1024, 1024)
        self.linear_sh = nn.Linear(1024, 1024)
        self.linear_at = nn.Linear(1024, 1024)
        self.fc_whole = nn.Linear(1024, classes[0])
        self.fc_hs = nn.Linear(1024, classes[1])
        self.fc_ub = nn.Linear(1024, classes[2])
        self.fc_lb = nn.Linear(1024, classes[3])
        self.fc_sh = nn.Linear(1024, classes[4])
        self.fc_at = nn.Linear(1024, classes[5])      
        
    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        
        attention_whole = self.avgpool(out)
        attention_whole = attention_whole.view(attention_whole.size(0), -1)
        attention_whole = self.linear_whole(attention_whole)
        attention_whole = F.sigmoid(attention_whole)
        attention_whole = attention_whole.view((-1, attention_whole.size(1), 1, 1))
        out_whole_c = out * attention_whole
        attention_whole = self.conv_whole(out)
        attention_whole = F.sigmoid(attention_whole)
        out_whole_s = out * attention_whole
        out_whole = out_whole_c + out_whole_s
        out_whole = self.avgpool(out_whole)
        out_whole = out_whole.view(out_whole.size(0), -1)
        out_whole = self.fc_whole(out_whole)
        
        attention_hs = self.avgpool(out)
        attention_hs = attention_hs.view(attention_hs.size(0), -1)
        attention_hs = self.linear_hs(attention_hs)
        attention_hs = F.sigmoid(attention_hs)
        attention_hs = attention_hs.view((-1, attention_hs.size(1), 1, 1))
        out_hs_c = out * attention_hs
        attention_hs = self.conv_hs(out)
        attention_hs = F.sigmoid(attention_hs)
        out_hs_s = out * attention_hs
        out_hs = out_hs_c + out_hs_s
        out_hs = self.avgpool(out_hs)
        out_hs = out_hs.view(out_hs.size(0), -1)
        out_hs = self.fc_hs(out_hs)
        
        attention_ub = self.avgpool(out)
        attention_ub = attention_ub.view(attention_ub.size(0), -1)
        attention_ub = self.linear_ub(attention_ub)
        attention_ub = F.sigmoid(attention_ub)
        attention_ub = attention_ub.view((-1, attention_ub.size(1), 1, 1))
        out_ub_c = out * attention_ub
        attention_ub = self.conv_ub(out)
        attention_ub = F.sigmoid(attention_ub)
        out_ub_s = out * attention_ub
        out_ub = out_ub_c + out_ub_s
        out_ub = self.avgpool(out_ub)
        out_ub = out_ub.view(out_ub.size(0), -1)
        out_ub = self.fc_ub(out_ub)
        
        attention_lb = self.avgpool(out)
        attention_lb = attention_lb.view(attention_lb.size(0), -1)
        attention_lb = self.linear_lb(attention_lb)
        attention_lb = F.sigmoid(attention_lb)
        attention_lb = attention_lb.view((-1, attention_lb.size(1), 1, 1))
        out_lb_c = out * attention_lb
        attention_lb = self.conv_lb(out)
        attention_lb = F.sigmoid(attention_lb)
        out_lb_s = out * attention_lb
        out_lb = out_lb_c + out_lb_s
        out_lb = self.avgpool(out_lb)
        out_lb = out_lb.view(out_lb.size(0), -1)
        out_lb = self.fc_lb(out_lb)
        
        attention_sh = self.avgpool(out)
        attention_sh = attention_sh.view(attention_sh.size(0), -1)
        attention_sh = self.linear_sh(attention_sh)
        attention_sh = F.sigmoid(attention_sh)
        attention_sh = attention_sh.view((-1, attention_sh.size(1), 1, 1))
        out_sh_c = out * attention_sh
        attention_sh = self.conv_sh(out)
        attention_sh = F.sigmoid(attention_sh)
        out_sh_s = out * attention_sh
        out_sh = out_sh_c + out_sh_s
        out_sh = self.avgpool(out_sh)
        out_sh = out_sh.view(out_sh.size(0), -1)
        out_sh = self.fc_sh(out_sh)
        
        attention_at = self.avgpool(out)
        attention_at = attention_at.view(attention_at.size(0), -1)
        attention_at = self.linear_at(attention_at)
        attention_at = F.sigmoid(attention_at)
        attention_at = attention_at.view((-1, attention_at.size(1), 1, 1))
        out_at_c = out * attention_at
        attention_at = self.conv_at(out)
        attention_at = F.sigmoid(attention_at)
        out_at_s = out * attention_at
        out_at = out_at_c + out_at_s
        out_at = self.avgpool(out_at)
        out_at = out_at.view(out_at.size(0), -1)
        out_at = self.fc_at(out_at)
        
        return F.sigmoid(torch.cat([out_whole, out_hs, out_ub, out_lb, out_sh, out_at], 1))    

if __name__ == "__main__":
    print("OK!")