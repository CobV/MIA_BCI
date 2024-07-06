import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
from torch import optim
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryF1Score, BinaryRecall


class ConvBlock1D(nn.Module):
    def __init__(self, params):
        super(ConvBlock1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=params["in_channels"],
                out_channels=params["out_channels"],
                kernel_size=params["kernel_size"],
                stride=params["stride"],
                padding=params["padding"],
                dilation=params["dilation"],
                groups=params["groups"]
            ),
            nn.BatchNorm1d(params["out_channels"]),
            nn.ReLU(),
            nn.Dropout(params['drop_prob']),
        )

    def forward(self, x):
        return self.net(x)

class Conv1DModel(nn.Module):
    def __init__(self, params, pooling_size):
        super(Conv1DModel, self).__init__()
        self.blocks = nn.Sequential(
            *[ConvBlock1D(i) for i in params])
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(pooling_size)
        #extra Norm 
        self.class_layer = nn.Linear(params[-1]["out_channels"]*pooling_size,1)

    def forward(self,x):
        x = self.blocks(x)  # (B,C,T)
        x = self.global_avg_pooling(x)
        B,C,T =  x.shape
        x = x.view(B,C*T)
        output = self.class_layer(x) 
        return output

#--------------------------------------------------------------------------
#from : https://gist.github.com/Timm877/43f13d5fcf0a6ddf06ec2b8384ea218f#file-eegnet_pytorch-py

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def get_temp_layer(filter_sizing_in,filter_sizing_out,receptive_field,drop_prob,nr):
    return nn.Sequential(
            nn.Conv2d(filter_sizing_in, filter_sizing_out, kernel_size=[1,receptive_field], stride=1, bias=False, groups=filter_sizing_in,
                padding='same'), 
            nn.BatchNorm2d(filter_sizing_out),
            nn.ReLU(True),
            nn.Dropout(drop_prob)
        )

class TestNetwork(nn.Module):
    def __init__(self, filter_sizing, drop_prob, receptive_field, pool_size, nr_temp_layers, divided):
        super(TestNetwork, self).__init__()
        self.channel_amount = 4 
        self.chunk_size = int(2.5 * 110) 
        self.num_classes = 1
        self.nr_temp_layers = nr_temp_layers
        self.drop_prob = drop_prob
        self.filter_sizing_per_layer = filter_sizing
        if divided:
            self.filter_sizing_per_layer = filter_sizing // nr_temp_layers
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, self.filter_sizing_per_layer, kernel_size=[1,1], stride=1, bias=False, groups=1, padding='same'), 
            nn.BatchNorm2d(self.filter_sizing_per_layer),
            nn.ReLU(True),
            nn.Dropout(drop_prob)
        )
        #------------------
        self.temporal_module = nn.Sequential(*[get_temp_layer(self.filter_sizing_per_layer, self.filter_sizing_per_layer,receptive_field,drop_prob,nr) for nr in range(self.nr_temp_layers)])

        self.spatial_module = nn.Sequential(
            nn.Conv2d(self.filter_sizing_per_layer,self.filter_sizing_per_layer, kernel_size=[self.channel_amount,1],bias=False,groups=self.filter_sizing_per_layer),
            nn.BatchNorm2d(self.filter_sizing_per_layer),
            nn.ELU(True),
            nn.Dropout(drop_prob)
        )
        #------------------
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(self.filter_sizing_per_layer, 1, kernel_size=1, stride=1, bias=False, padding='same'), 
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Dropout(drop_prob)
        )
        
        self.avgpool = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)
        self.fc = nn.Linear(self.feature_dim(), self.num_classes)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.channel_amount, self.chunk_size)
            out = self.input_layer(mock_eeg)
            out = self.temporal_module(out)
            out = self.spatial_module(out)
            out = self.output_layer(out)
            out = self.avgpool(out)
            out = torch.flatten(out,start_dim=1)
        return out.shape[-1]

    def forward(self,x,targets=None):
        x = x.unsqueeze(1)
        out = self.input_layer(x)
        out = self.temporal_module(out)
        out = self.spatial_module(out)
        out = self.output_layer(out)
        out = self.avgpool(out)
        out = torch.flatten(out,start_dim=1)
        out = self.fc(out)
        return out

class EEGNet2TL(nn.Module):
    def __init__(self, filter_sizing, drop_prob, D,  receptive_field, separable, sep_kernel_size, pool_kernel_size2, pooling_overlap, nr_temp_layers, pooling_layer, spatial_layer):
        super(EEGNet2TL, self).__init__()
        self.channel_amount = 4 
        self.chunk_size = int(2.5 * 110) #window_duration = 2.5; downsampled_freq = 110
        self.num_classes = 1
        self.nr_temp_layers = nr_temp_layers
        self.pooling = pooling_layer
        self.spatial_layer = spatial_layer
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filter_sizing, kernel_size=[1,receptive_field], stride=1, bias=False,
                padding='same'), 
            nn.BatchNorm2d(filter_sizing),
            nn.Dropout(drop_prob)
        )
        self.temporal1=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing, kernel_size=[1,receptive_field//2], stride=1, bias=False,
                padding='same'), 
            nn.BatchNorm2d(filter_sizing),
            nn.Dropout(drop_prob)
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing*D, kernel_size=[self.channel_amount,1],bias=False,\
                groups=filter_sizing),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
            nn.Dropout(drop_prob)
        )
        self.sep_kernel_size = sep_kernel_size
        self.add_seperable = separable

        self.seperable=nn.Sequential(
            nn.Conv2d(filter_sizing*D,filter_sizing*D, kernel_size=[1,self.sep_kernel_size],\
                padding='same',groups=filter_sizing*D, bias=False),
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,1], padding='same',groups=1, bias=False),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
            nn.Dropout(drop_prob)
        )
        self.pool_kernel_size2 = pool_kernel_size2
        self.pooling_overlap = pooling_overlap
        self.avgpool1 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)
        self.avgpool2 = nn.AvgPool2d([1, self.pool_kernel_size2], stride=[1, self.pool_kernel_size2 // self.pooling_overlap], padding=0)
        self.view = nn.Sequential(Flatten())
        self.fc2 = nn.Linear(self.feature_dim(), self.num_classes)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.channel_amount, self.chunk_size)
            mock_eeg = self.temporal(mock_eeg)
            if self.nr_temp_layers == 2:
                mock_eeg = self.temporal1(mock_eeg)
            if self.spatial_layer == True:
                mock_eeg = self.spatial(mock_eeg)
            if self.pooling == True:
                mock_eeg = self.avgpool1(mock_eeg)
            if self.add_seperable:
                mock_eeg = self.seperable(mock_eeg)
            mock_eeg = self.avgpool2(mock_eeg)
            result = 1
            for value in mock_eeg.shape[1:]:
                result *= value
        return result

    def forward(self,x,targets=None):
        x = x.unsqueeze(1)
        out = self.temporal(x)
        if self.nr_temp_layers == 2:
            out = self.temporal1(out)
        if self.spatial_layer == True:
            out = self.spatial(out)
        if self.pooling == True:
            out = self.avgpool1(out)
        if self.add_seperable:
            out = self.seperable(out)
        out = self.avgpool2(out)
        out = out.view(out.size(0), -1)
        logits = self.fc2(out)
        return logits
    
class LightWrapper(L.LightningModule):
    def __init__(self, module, lr_rate, weight_decay, enable_dp = False, **kwargs):
        super().__init__()
        self.module = module
        self.lr_rate = lr_rate
        self.weight_decay = weight_decay


    def training_step(self, batch, batch_idx):
        obj, target = batch
        preds = self.module(obj).flatten()
        loss = F.binary_cross_entropy_with_logits(preds, target)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        obj, target = batch
        preds = self.module(obj).flatten()
        loss = F.binary_cross_entropy_with_logits(preds, target)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.module(x).flatten()
        preds = F.sigmoid(preds)
        recall = BinaryRecall().to(self.device)
        f1_score = BinaryF1Score().to(self.device)
        precision = BinaryPrecision().to(self.device)
        accuracy = BinaryAccuracy().to(self.device)
        rec = recall(preds, y)
        f1 = f1_score(preds, y)
        acc = accuracy(preds, y)
        pr = precision(preds, y)
        self.log('test_acc',acc, on_epoch=True, logger=True)
        self.log('test_f1',f1, on_epoch=True, logger=True)
        self.log('test_precision',pr, on_epoch=True, logger=True)
        self.log('test_recall',rec, on_epoch=True, logger=True)
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr_rate, weight_decay=self.weight_decay)
        return optimizer
