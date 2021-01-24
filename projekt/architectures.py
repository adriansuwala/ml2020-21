import torch

from torch import nn

class EncoderDecoderGenerator(nn.Module):
    def __init__(self, optim, lambd, bnorm_track_stats=False, explicit_instancenorm=False, **kwargs):
        super(EncoderDecoderGenerator, self).__init__()

        normalization = nn.BatchNorm2d if not explicit_instancenorm else nn.InstanceNorm2d

        # encoder
        # C64-C128-C256-C512-C512-C512-C512-C512
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True), # no batch norm in the first layer

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            normalization(128, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            normalization(256, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )

        # decoder
        # CD512-CD512-CD512-C512-C256-C128-C64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.Dropout(0.5),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.Dropout(0.5),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.Dropout(0.5),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            normalization(256, track_running_stats=bnorm_track_stats),
            nn.ReLU(True),
       
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            normalization(128, track_running_stats=bnorm_track_stats),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            normalization(64, track_running_stats=bnorm_track_stats),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.optim = optim(self.parameters(), **kwargs)
    
        # All weights initialized from normal with mean 0 and std 0.02
        for layer in self.encoder:
            if type(layer) != nn.LeakyReLU:
                torch.nn.init.normal_(layer.weight, 0, 0.02)
        for layer in self.decoder:
            if type(layer) not in [nn.ReLU, nn.Dropout, nn.Tanh]:
                torch.nn.init.normal_(layer.weight, 0, 0.02)
        
        self.l1 = torch.nn.L1Loss()
        self.lambd = lambd

    def eval(self):
        """Uses dropout during testing as a source of noise and nondeterminism."""
        for module in self.children():
            module.train(type(module) == nn.Dropout)

    def forward(self, y):
        x = self.encoder(y)
        x = self.decoder(x)
        return x
    
    def loss(self, dis, y, target):
        x = self(y)
        p = dis(x, y)
        if self.lambd == 0:
            return dis.loss(p, target)
        else:
            return dis.loss(p, target) + self.lambd * self.l1(p, target)

    def update(self, dis, y, device="cpu"):
        self.optim.zero_grad()
        loss = self.loss(dis, y, torch.ones(y.shape[0], 1).to(device))
        loss.backward()
        self.optim.step()
        return loss


class UNetGenerator(nn.Module):
    def __init__(self, optim, lambd, bnorm_track_stats=False, explicit_instancenorm=False, **kwargs):
        super(UNetGenerator, self).__init__()

        normalization = nn.BatchNorm2d if not explicit_instancenorm else nn.InstanceNorm2d

        # encoder
        # C64-C128-C256-C512-C512-C512-C512-C512
        self.en_1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True) # no batch norm in the first layer
        )
        self.en_2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            normalization(128, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True)
        )
        self.en_3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            normalization(256, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True)
        )
        self.en_4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True)
        )
        self.en_5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True)
        )
        self.en_6 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True)
        )
        self.en_7 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.LeakyReLU(0.2, True)
        )
        self.en_8 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )

        # decoder
        # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128 
        # For some reason there is one more layer in the U-net decoder in the paper
        # which causes (including last convolution with Tanh) to output 512x512 images
        # instead of 256x256
        # hence, I removed (for now) one C1024
        # there is also inconsistency with notation in the paper
        # since they first define Ck as Conv-BN-ReLU with k filters
        # and then use it in U-net decoder like it's k _channels_
        self.de_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.Dropout(0.5),
            nn.ReLU(True)
        )
        self.de_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.Dropout(0.5),
            nn.ReLU(True)
        )
        self.de_3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.Dropout(0.5),
            nn.ReLU(True)
        )
        self.de_4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            normalization(512, track_running_stats=bnorm_track_stats),
            nn.ReLU(True)
        )
        self.de_5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1),
            normalization(256, track_running_stats=bnorm_track_stats),
            nn.ReLU(True)
        )
        self.de_6 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1),
            normalization(128, track_running_stats=bnorm_track_stats),
            nn.ReLU(True)
        )
        self.de_7 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            normalization(64, track_running_stats=bnorm_track_stats),
            nn.ReLU(True)
        )
        self.de_8 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )
        self.optim = optim(self.parameters(), **kwargs)
    
        for block in [self.en_1, self.en_2, self.en_3, self.en_4, self.en_5, self.en_6, self.en_7, self.en_8]:
            for layer in block:
                if type(layer) not in [nn.LeakyReLU]:
                    torch.nn.init.normal_(layer.weight, 0, 0.02)
        for block in [self.de_1, self.de_2, self.de_3, self.de_4, self.de_5, self.de_6, self.de_7, self.de_8]:
            for layer in block:
                if type(layer) not in [nn.ReLU, nn.Dropout, nn.Tanh]:
                    torch.nn.init.normal_(layer.weight, 0, 0.02)
        
        self.l1 = torch.nn.L1Loss()
        self.lambd = lambd
    
    def eval(self):
        """Uses dropout during testing as a source of noise and nondeterminism."""
        for module in self.children():
            module.train(type(module) == nn.Dropout)

    def forward(self, y):
        x1 = self.en_1(y)
        x2 = self.en_2(x1)
        x3 = self.en_3(x2)
        x4 = self.en_4(x3)
        x5 = self.en_5(x4)
        x6 = self.en_6(x5)
        x7 = self.en_7(x6)
        x8 = self.en_8(x7)

        z = self.de_1(x8)
        z = self.de_2(torch.cat((z, x7), dim=1))
        z = self.de_3(torch.cat((z, x6), dim=1))
        z = self.de_4(torch.cat((z, x5), dim=1))
        z = self.de_5(torch.cat((z, x4), dim=1))
        z = self.de_6(torch.cat((z, x3), dim=1))
        z = self.de_7(torch.cat((z, x2), dim=1))
        z = self.de_8(torch.cat((z, x1), dim=1))
        return z
    
    def loss(self, dis, y, target):
        x = self(y)
        p = dis(x, y)
        if self.lambd == 0:
            return dis.loss(p, target)
        else:
            return dis.loss(p, target) + self.lambd * self.l1(p, target)

    def update(self, dis, y, device="cpu"):
        self.optim.zero_grad()
        loss = self.loss(dis, y, torch.ones(y.shape[0], 1).to(device))
        loss.backward()
        self.optim.step()
        return loss


class PatchGANDiscriminator(nn.Module):
    def __init__(self, optim, criterion, **kwargs):
        super(PatchGANDiscriminator, self).__init__()
        self.criterion = criterion
        
        # 70x70
        # C64-C128-C256-C512
        # formula for receptive field: (output - 1) * stride + kernel
        # compute that for every layer starting with last one
        # (receptive field for next layer is output for previous layer)
        
        # 6 input channels because mask is concatenated (no idea if this is right,
        # just an idea since there is nothing about what to do with conditional input in the paper)
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.optim = optim(self.parameters(), **kwargs)

        for layer in self.model:
            if type(layer) not in [nn.LeakyReLU, nn.Sigmoid]:
                nn.init.normal_(layer.weight, 0, 0.02)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.model(x)
        x = torch.mean(x, dim=(2,3))
        return x

    def loss(self, x, y):
        return self.criterion(x, y)

    def update(self, true_sample, gen_sample, cond, device="cpu"):
        self.optim.zero_grad()

        true_loss = self.loss(self(true_sample, cond), torch.ones(true_sample.shape[0], 1).to(device))
        prob = self(gen_sample, cond)
        false_loss = self.loss(prob, torch.zeros(gen_sample.shape[0], 1).to(device))

        # division by 2 to slow down D
        total_loss = (true_loss + false_loss) / 2
        total_loss.backward()
        self.optim.step()
        return total_loss, prob

