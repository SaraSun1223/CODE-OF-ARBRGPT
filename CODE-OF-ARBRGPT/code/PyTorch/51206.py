import time



from torch import nn

import torch

from torch.cuda.amp import GradScaler, autocast

from torch.nn import functional

from torch.optim import SGD





class Network(nn.Module):

    def __init__(self, cast_for_upsample=False):

        super().__init__()

        self.cast_for_upsample = cast_for_upsample



        self.layers = nn.Sequential(

            nn.Conv3d(1, 32, 3, 1, 1, 1, 1, False),

            nn.LeakyReLU(1e-2, True),

            nn.Conv3d(32, 32, 3, 1, 1, 1, 1, False),

            nn.LeakyReLU(1e-2, True),



            nn.Conv3d(32, 64, 3, 2, 1, 1, 1, False),

            nn.LeakyReLU(1e-2, True),

            nn.Conv3d(64, 64, 3, 1, 1, 1, 1, False),

            nn.LeakyReLU(1e-2, True),



            nn.Conv3d(64, 128, 3, 2, 1, 1, 1, False),

            nn.LeakyReLU(1e-2, True),

            nn.Conv3d(128, 128, 3, 1, 1, 1, 1, False),

            nn.LeakyReLU(1e-2, True),



            nn.Conv3d(128, 256, 3, 2, 1, 1, 1, False),

            nn.LeakyReLU(1e-2, True),

            nn.Conv3d(256, 256, 3, 1, 1, 1, 1, False),

            nn.LeakyReLU(1e-2, True),



            nn.Conv3d(256, 512, 3, 2, 1, 1, 1, False),

            nn.LeakyReLU(1e-2, True),

            nn.Conv3d(512, 512, 3, 1, 1, 1, 1, False),

        )



    def forward(self, x):

        down = self.layers(x)

        if self.cast_for_upsample:

            up = nn.functional.interpolate(down.float(), x.shape[2:], None, 'trilinear').half()

        else:

            up = nn.functional.interpolate(down, x.shape[2:], None, 'trilinear')

        return up





if __name__ == "__main__":

    inp = torch.rand((2, 1, 64, 64, 64)).cuda()



    net = Network(cast_for_upsample=False).cuda()

    optimizer = SGD(net.parameters(), 0.001)



    torch.cuda.empty_cache()



    # warmup

    for _ in range(10):

        optimizer.zero_grad()

        out = net(inp)

        l = torch.square(inp - out).mean() # just the MSE between input and output as a dummy loss function

        l.backward()

        optimizer.step()



    # fp32 measurement

    st = time()

    for _ in range(100):

        optimizer.zero_grad()

        out = net(inp)

        l = torch.square(inp - out).mean() # just the MSE between input and output as a dummy loss function

        l.backward()

        optimizer.step()

    print('fp32:', time() - st)



    ####################################################

    # now AMP

    net = Network(cast_for_upsample=False).cuda()

    optimizer = SGD(net.parameters(), 0.001)

    scaler = GradScaler()



    torch.cuda.empty_cache()



    # warmup

    for _ in range(10):

        optimizer.zero_grad()



        with autocast():

            out = net(inp)

            l = torch.square(inp - out).mean()  # just the MSE between input and output as a dummy loss function



        scaler.scale(l).backward()

        scaler.step(optimizer)

        scaler.update()



    # amp measurement

    st = time()

    for _ in range(100):

        optimizer.zero_grad()



        with autocast():

            out = net(inp)

            l = torch.square(inp - out).mean()  # just the MSE between input and output as a dummy loss function



        scaler.scale(l).backward()

        scaler.step(optimizer)

        scaler.update()

    print('amp:', time() - st)



    ####################################################

    # now AMP with hacking interpolate so that is runs in fp32

    net = Network(cast_for_upsample=True).cuda()

    optimizer = SGD(net.parameters(), 0.001)

    scaler = GradScaler()



    torch.cuda.empty_cache()



    # warmup

    for _ in range(10):

        optimizer.zero_grad()



        with autocast():

            out = net(inp)

            l = torch.square(inp - out).mean()  # just the MSE between input and output as a dummy loss function



        scaler.scale(l).backward()

        scaler.step(optimizer)

        scaler.update()



    # amp measurement

    st = time()

    for _ in range(100):

        optimizer.zero_grad()



        with autocast():

            out = net(inp)

            l = torch.square(inp - out).mean()  # just the MSE between input and output as a dummy loss function



        scaler.scale(l).backward()

        scaler.step(optimizer)

        scaler.update()

    print('amp cast to float:', time() - st)