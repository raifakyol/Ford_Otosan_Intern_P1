import torch
import torch.nn as nn


def double_conv(in_channels,out_channels,mid_channels=None):#double_conv adında 2 parametresi olan bir fonksiyon oluşturuldu
    if not mid_channels:
        mid_channels = out_channels
        return nn.Sequential(
        #nn.Sequential sinir ağı oluştumamızı sağlar
        
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,padding=1),
        #conv2D, 2D veriye (örneğin bir görüntü) evrişimin bir işlevidir.
        #kernel_size = 2D evrişim penceresinin genişliğini ve yüksekliğini belirten 2 tuple.
        #kernel_size (int veya tuple) - Convolution  çekirdeğin boyutu
        #padding (int veya tuple, isteğe bağlı) - Girişin her iki tarafına sıfır dolgu eklendi (Varsayılan: 0)
        #in_channels 3 channels (renkli görüntüler) görüntüler için başlangıçta 3'tür.
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),#activation işlevi
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
        #Sinir ağlarını yeniden merkezliyerek ve yeniden ölçeklendirilerek giriş katmanı normalleştirildi
            nn.ReLU(inplace=True),
       
        
    )

class FoInternNet(nn.Module):#FoInternNet adında bir sınıf oluşturuldu
    def __init__(self,input_size,n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)  
        
        #self.dropout=nn.Dropout2d(0.5)
        self.maxpool = nn.MaxPool2d(2)#Filtrenin hareket ettiği piksellerdeki değerlerin maksimumunu alır.
        #kernel_size = "pooling" yapılacak alanı adım adım belirler.
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        #Görüntü boyutlarını artırmak istediğimizde, temel olarak bir görüntüyü genişletir 
        #ve orijinal görüntünün satır ve sütunlarındaki "boşlukları" doldururuz.
        #scale_factor: yukarı veya aşağı örneklemek için ölçek faktörü.
        #Bilinear: Doğrusal enterpolasyonlar kullanarak pikselin değerini hesaplamak için yakındaki tüm pikselleri kullanır.
        #align_corners = Doğru, piksel noktaları kılavuz olarak kabul edilir. Köşelerdeki noktalar hizalanır.
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_classes, 1)
         
             
    def forward(self, x):
        #print(x.shape)
        conv1 = self.dconv_down1(x)
        #print(conv1.shape)
        
        x = self.maxpool(conv1)
        #x=self.dropout(x)
        #print("maxpool")
        #print(x.shape)
        
        conv2 = self.dconv_down2(x)
        #print(conv2.shape)
        
        x = self.maxpool(conv2)
        #x=self.dropout(x)
        #print("maxpool")
        #print(x.shape)
        
        conv3 = self.dconv_down3(x) 
        #print(conv3.shape)
        
        x = self.maxpool(conv3)   
        #x=self.dropout(x)
        #print("maxpool")
        #print(x.shape)
        
        x = self.dconv_down4(x)
        #print(x.shape)
        
        x = self.upsample(x)    
        #print("upsample")
        #print(x.shape)
        
        x = torch.cat([x, conv3], dim=1)
        #Verilen tensör dizisini verilen boyutta birleştirir 
        #print("cat")
        #print(x.shape)
        
        x = self.dconv_up3(x)
        #print(x.shape)

        x = self.upsample(x)    
        #print("upsample")
        #print(x.shape)

        x = torch.cat([x, conv2], dim=1)    
        #print("cat")
        #print(x.shape)

        x = self.dconv_up2(x)
        #print(x.shape)
        
        x = self.upsample(x)    
        #print("upsample")
        #print(x.shape)

        x = torch.cat([x, conv1], dim=1)   
        #print("cat")
        #print(x.shape)
      
        x = self.dconv_up1(x)
        #print(x.shape)
       
        x = self.conv_last(x)
        #print(x.shape)
      
        x = nn.Softmax(dim=1)(x)
        #print(x.shape)

        return x