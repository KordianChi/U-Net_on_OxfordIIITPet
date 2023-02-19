from torch.nn import Module
from torch.nn import Sequential

from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import ConvTranspose2d

from torch import cat



def double_conv_block(channel_in, channel_out):
    
    conv_block = Sequential(
        Conv2d(channel_in, channel_out, 3, 1 ,1),
        ReLU(),
        BatchNorm2d(channel_out)
        )
    return conv_block
    




def decoder_deconv_block(channel_in, channel_out):
    
    deconv_block = Sequential(
        ConvTranspose2d(channel_in, channel_out, 2, 2)
        )
    return deconv_block
    
    
class UNet(Module):
    
    def __init__(self):
        super(UNet, self).__init__()
    
        self.encoder_double_conv_block_1 = double_conv_block(3, 64)
        self.encoder_double_conv_block_2 = double_conv_block(64, 128)
        self.encoder_double_conv_block_3 = double_conv_block(128, 256)
        self.encoder_double_conv_block_4 = double_conv_block(256, 512)
        self.encoder_double_conv_block_5 = double_conv_block(512, 1024)
        
        self.max_pool_2d = MaxPool2d(kernel_size=2, stride=2)
        
        self.decoder_conv_block_1 = double_conv_block(1024, 512)
        self.decoder_conv_block_2 = double_conv_block(512, 256)
        self.decoder_conv_block_3 = double_conv_block(256, 128)
        self.decoder_conv_block_4 = double_conv_block(128, 64)
        
        self.decoder_deconv_block_1 = decoder_deconv_block(1024, 512)
        self.decoder_deconv_block_2 = decoder_deconv_block(512, 256)
        self.decoder_deconv_block_3 = decoder_deconv_block(256, 128)
        self.decoder_deconv_block_4 = decoder_deconv_block(128, 64)
        
        self.output_conv = Conv2d(64, 1, kernel_size=1)
        
    def forward(self, image):
        
        # encoder step
        
        out_1 = self.encoder_double_conv_block_1(image)
        out_2 = self.max_pool_2d(out_1)
        
        out_3 = self.encoder_double_conv_block_2(out_2)
        out_4 = self.max_pool_2d(out_3)
        
        out_5 = self.encoder_double_conv_block_3(out_4)
        out_6 = self.max_pool_2d(out_5)
        
        out_7 = self.encoder_double_conv_block_4(out_6)
        out_8 = self.max_pool_2d(out_7)
        
        out_9 = self.encoder_double_conv_block_5(out_8)
        
        # decoder step
        
        out_10 = self.decoder_deconv_block_1(out_9)
        out_11 = cat((out_10, out_7), dim=1)
        out_12 = self.decoder_conv_block_1(out_11)
    
        out_13 = self.decoder_deconv_block_2(out_12)
        out_14 = cat((out_13, out_5), dim=1)
        out_15 = self.decoder_conv_block_2(out_14)
        
        out_16 = self.decoder_deconv_block_3(out_15)
        out_17 = cat((out_16, out_3), dim=1)
        out_18 = self.decoder_conv_block_3(out_17)
        
        out_19 = self.decoder_deconv_block_4(out_18)
        out_20 = cat((out_19, out_1), dim=1)
        out_21 = self.decoder_conv_block_4(out_20)
        
        final_out = self.output_conv(out_21)
    
        return final_out
        
        
        
        
        
        
        