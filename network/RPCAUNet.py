# -*- coding: utf-8 -*-
"""

@reference: Deep Unfolded Robust PCA With Application to Clutter Suppression in Ultrasound. https://github.com/KrakenLeaf/CORONA
@reference: Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. https://github.com/automan000/Convolutional_LSTM_PyTorch

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import math


def to_var(X, CalInGPU):
    if CalInGPU and torch.cuda.is_available():
        X = X.cuda()
    return Variable(X)


# CLSTM module
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, CalInGPU):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.CalInGPU = CalInGPU

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = to_var(nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])), self.CalInGPU)
            self.Wcf = to_var(nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])), self.CalInGPU)
            self.Wco = to_var(nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])), self.CalInGPU)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (to_var(torch.zeros(batch_size, hidden, shape[0], shape[1]), self.CalInGPU),
                to_var(torch.zeros(batch_size, hidden, shape[0], shape[1]), self.CalInGPU))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], CalInGPU=True):
        super(ConvLSTM, self).__init__()
        self.CalInGPU = CalInGPU
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.CalInGPU)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


# resnet module
class ConvolutionalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # layers list
        layers = list()

        # conv layer
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # BN layer
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # activation layer
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        output = self.conv_block(input)

        return output


class SubPixelConvolutionalBlock(nn.Module):
    """
    SubPixelConvolutionalBlock, including convolution, SubPixelConvolutional layer, activation layer
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        super(SubPixelConvolutionalBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.pixel_shuffle(output)
        output = self.prelu(output)

        return output


class ResidualBlock(nn.Module):

    def __init__(self, kernel_size=3, n_channels=64):
        super(ResidualBlock, self).__init__()

        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        residual = input
        output = self.conv_block1(input)
        output = self.conv_block2(output)
        output = output + residual

        return output


class SRResNet(nn.Module):
    """
    SRResNet module
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16,
                 scaling_factor=4, CalInGPU=True):
        super(SRResNet, self).__init__()
        self.CalInGPU = CalInGPU

        # scaling_factor must be 2, 4 or 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}

        self.conv_block1 = ConvolutionalBlock(in_channels=1, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)
        self.lstm = ConvLSTM(input_channels=16, hidden_channels=[32, 32, 16], kernel_size=3, step=3,
                             effective_step=[2], CalInGPU=self.CalInGPU)
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=1, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = self.lstm(output)
        output = output[0][0]
        output = output + residual
        output = self.subpixel_convolutional_blocks(output)
        sr_imgs = self.conv_block3(output)

        return sr_imgs


# Unfolded RPCA module
class Conv3dC(nn.Module):
    def __init__(self, kernel):
        super(Conv3dC, self).__init__()

        pad0 = int((kernel[0] - 1) / 2)
        self.convR = nn.Conv2d(1, 1, kernel[0], stride=1, padding=pad0)
        self.convI = nn.Conv2d(1, 1, kernel[0], stride=1, padding=pad0)

    def forward(self, x):
        n = x.shape[-1]
        nh = int(n / 2)

        xR = x[None, :, :, 0:n].permute(3, 0, 1, 2)
        xR = self.convR(xR)
        x = xR.squeeze(1)
        x = x.permute(1, 2, 0)
        return x


class ISTACell(nn.Module):
    '''
    each layer of the RPCA-UNet
    '''

    def __init__(self, kernel, exp_L, exp_S, coef_L, coef_S, CalInGPU):
        super(ISTACell, self).__init__()

        self.conv1 = Conv3dC(kernel)
        self.conv2 = Conv3dC(kernel)
        self.conv3 = Conv3dC(kernel)
        self.conv4 = Conv3dC(kernel)
        self.conv5 = Conv3dC(kernel)
        self.conv6 = Conv3dC(kernel)

        self.exp_L = nn.Parameter(exp_L)
        self.exp_S = nn.Parameter(exp_S)

        self.coef_L = coef_L
        self.coef_S = coef_S
        self.CalInGPU = CalInGPU
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.sr_module1 = SRResNet(large_kernel_size=9, small_kernel_size=3,
                                  n_channels=16, n_blocks=4, scaling_factor=2, CalInGPU=self.CalInGPU)
        self.sr_module2 = SRResNet(large_kernel_size=9, small_kernel_size=3,
                                  n_channels=16, n_blocks=4, scaling_factor=2, CalInGPU=self.CalInGPU)

    def forward(self, data):
        # down sampling
        data_tmp = data.permute(3, 0, 1, 2)
        data_tmp = nn.functional.interpolate(data_tmp, scale_factor=0.5)
        data_tmp = data_tmp.permute(1, 2, 3, 0)
        x = data_tmp[0]
        L = data_tmp[1]
        S = data_tmp[2]

        H, W, T2 = x.shape

        Ltmp = self.conv1(x) + self.conv2(L) + self.conv3(S)
        Stmp = self.conv4(x) + self.conv5(L) + self.conv6(S)

        thL = self.sig(self.exp_L) * self.coef_L
        thS = self.sig(self.exp_S) * self.coef_S

        L = self.svtC(Ltmp.view(H * W, T2), thL)
        S = self.mixthre(Stmp.view(H * W, T2), thS)
        L = L.view(H, W, T2)
        S = S.view(H, W, T2)

        # superresolution for L and S
        L = L[None, :, :, :].permute(3, 0, 1, 2)
        L = self.sr_module1(L)
        L = L.squeeze(1)
        L = L.permute(1, 2, 0)
        S = S[None, :, :, :].permute(3, 0, 1, 2)
        S = self.sr_module2(S)
        S = S.squeeze(1)
        S = S.permute(1, 2, 0)

        data[1] = L
        data[2] = S

        return data

    def svtC(self, x, th):
        m, n = x.shape
        U, S, V = torch.svd(x)
        S = self.relu(S - th * S[0])
        US = to_var(torch.zeros(m, n), self.CalInGPU)
        stmp = to_var(torch.zeros(n), self.CalInGPU)
        stmp[0:S.shape[0]] = S
        minmn = min(m, n)
        US[:, 0:minmn] = U[:, 0:minmn]

        x = (US * stmp) @ V.t()
        return x

    def mixthre(self, x, th):
        n = x.shape[-1]
        nh = int(n / 2)
        xR, xI = x[:, 0:nh], x[:, nh:n]
        normx = xR ** 2 + xI ** 2
        normx = torch.cat((normx, normx), -1)
        x = self.relu((1 - th * torch.mean(normx) / normx)) * x

        return x


class RPCAUNet(nn.Module):
    def __init__(self, params=None):
        super(RPCAUNet, self).__init__()

        self.layers = params['layers']
        self.kernel = params['kernel']
        self.CalInGPU = params['CalInGPU']
        self.coef_L = to_var(torch.tensor(params['coef_L'], dtype=torch.float),
                             self.CalInGPU)
        self.coef_S = to_var(torch.tensor(params['coef_S'], dtype=torch.float),
                             self.CalInGPU)
        self.exp_L = to_var(torch.zeros(self.layers, requires_grad=True),
                            self.CalInGPU)
        self.exp_S = to_var(torch.zeros(self.layers, requires_grad=True),
                            self.CalInGPU)
        self.sig = nn.Sigmoid()

        self.relu = nn.ReLU()

        self.filter = self.makelayers()

    def makelayers(self):
        filt = []
        for i in range(self.layers):
            filt.append(ISTACell(self.kernel[i], self.exp_L[i], self.exp_S[i],
                                 self.coef_L, self.coef_S, self.CalInGPU))

        return nn.Sequential(*filt)

    def forward(self, x):
        data = to_var(torch.zeros([3] + list(x.shape)), self.CalInGPU)
        data[0] = x

        data = self.filter(data)
        L = data[1]
        S = data[2]

        return L, S

    def getexp_LS(self):
        exp_L, exp_S = self.sig(self.exp_L) * self.coef_L, self.sig(self.exp_S) * self.coef_S
        if torch.cuda.is_available():
            exp_L = exp_L.cpu().detach().numpy()
            exp_S = exp_S.cpu().detach().numpy()
        else:
            exp_L = exp_L.detach().numpy()
            exp_S = exp_S.detach().numpy()

        return exp_L, exp_S


if __name__ == '__main__':
    params_net = {}
    params_net['layers'] = 4
    params_net['kernel'] = [(5, 1)] * 2 + [(3, 1)] * 2
    params_net['coef_L'] = 0.4
    params_net['coef_S'] = 1.8
    params_net['CalInGPU'] = True  # whether to calculate in GPU
    params_net['kernel'] = params_net['kernel'][0:params_net['layers']]
    net = RPCAUNet(params_net).cuda()
    data = torch.rand([64, 64, 20]).cuda()
    L, S = net(data)
    print(L.shape)
    print(S.shape)
