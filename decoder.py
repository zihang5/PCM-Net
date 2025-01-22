import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class RegHead(nn.Module):
    def __init__(self, in_channels):
        super(RegHead, self).__init__()
        self.conv3 = nn.Conv3d(in_channels, 3, 3, 1, 1)
        self.conv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv3.weight.shape))
        self.conv3.bias = nn.Parameter(torch.zeros(self.conv3.bias.shape))

    def forward(self, x):
        x = self.conv3(x)
        return x
        
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)
    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width, depth = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width, depth)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width, depth)
    return x

class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = 'relu'
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(self.in_channels, self.in_channels, kernel_size, 1, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm3d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            x = x+dw_out
        return outputs

class MDCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1,3,5]):
        super(MDCB, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = 2
        self.activation = 'relu'
        self.n_scales = len(self.kernel_sizes)
        self.use_skip_connection = True

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes)
        self.pconv2 = nn.Sequential(
            nn.Conv3d(self.ex_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(self.out_channels),
        )
    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        dout = 0
        for dwout in msdc_outs:
            dout = dout + dwout
        dout = channel_shuffle(dout, gcd(self.ex_channels,self.out_channels))
        out = self.pconv2(dout)
        return x + out

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = 3,
            padding=1,
            stride=1,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        nm = nn.BatchNorm3d(out_channels)
        relu = nn.LeakyReLU(inplace=True)
        super(Conv3dReLU, self).__init__(conv, nm, relu)

class ACRG(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(ACRG, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv3d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv3d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.reweight = MLP(self.in_channels, self.in_channels//4, self.in_channels*2)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        
        a = (max_out+avg_out).flatten(2).squeeze(2)
        a = self.reweight(a).reshape(2, a.shape[1], 1).permute(2, 0, 1).softmax(dim=2).unsqueeze(2).unsqueeze(2).unsqueeze(2).view(2,1, a.shape[1], 1,1,1)
        out = a[0]*avg_out + a[1]*max_out + avg_out + max_out
        return self.sigmoid(out)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """
    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class Convup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(Convup, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        kernel_sizes = [1,3,5]
        self.mdcb1 = MDCB(8, 8, kernel_sizes=kernel_sizes)
        self.mdcb2 = MDCB(16, 16,  kernel_sizes=kernel_sizes)
        self.mdcb3 = MDCB(32, 32, kernel_sizes=kernel_sizes)
        self.mdcb4 = MDCB(64, 64,  kernel_sizes=kernel_sizes)
        self.mdcb5 = MDCB(128, 128, kernel_sizes=kernel_sizes)

        self.reg5 = RegHead(64)
        self.reg4 = RegHead(32)
        self.reg3 = RegHead(16)
        self.reg2 = RegHead(8)
        self.reg1 = RegHead(4)

        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conva1 = Conv3dReLU(256, 128)
        self.conva2 = Conv3dReLU(128, 64)
        self.conva3 = Convup(128, 64)
        
        self.convb1 = Conv3dReLU(192, 64)
        self.convb2 = Conv3dReLU(64, 32)
        self.convb3 = Convup(64, 32)
        
        self.convc1 = Conv3dReLU(96, 32)
        self.convc2 = Conv3dReLU(32, 16)
        self.convc3 = Convup(32, 16)

        self.convd1 = Conv3dReLU(48, 16)
        self.convd2 = Conv3dReLU(16, 8)
        self.convd3 = Convup(16, 8)

        self.conve1 = Conv3dReLU(24, 8)
        self.conve2 = Conv3dReLU(8, 4)
        inshape = [160,192,160]

        self.integrate = nn.ModuleList()
        self.transformer = nn.ModuleList()
        for i in range(5):
            self.transformer.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            self.integrate.append(VecInt([s // 2 ** i for s in inshape], nsteps=7))
        
        
        self.acrg5 = ACRG(128)
        self.acrg4 = ACRG(64)
        self.acrg3 = ACRG(32)
        self.acrg2 = ACRG(16)
        self.acrg1 = ACRG(8)

    def forward(self,l):
        #torch.Size([1, 8, 160, 192, 160]) torch.Size([1, 16, 80, 96, 80]) torch.Size([1, 32, 40, 48, 40]) torch.Size([1, 64, 20, 24, 20]) torch.Size([1, 128, 10, 12, 10])
        x5, x4, x3, x2, x1, y5, y4, y3, y2, y1 = l
        d5 = torch.cat([x5, y5],dim = 1)#torch.Size([1, 384, 10, 12, 10])
        d5 = self.conva1(d5)
        d5 = self.acrg5(d5)*d5
        d5 = self.mdcb5(d5)
        d51 = self.conva2(d5)
        w = self.reg5(d51)
        d5 = self.conva3(d5)
        w = self.integrate[4](w)
        flow = self.upsample_trilin(2*w)

        x45t = self.transformer[3](x4, flow)
        d4 = torch.cat([x45t, y4, d5],dim = 1)
        d4 = self.convb1(d4)
        d4 = self.acrg4(d4)*d4
        d4 = self.mdcb4(d4)
        d41 = self.convb2(d4)
        w = self.reg4(d41)
        d4 = self.convb3(d4)
        w = self.integrate[3](w)
        flow = self.upsample_trilin(2*(self.transformer[3](flow, w)+w))

        x34t = self.transformer[2](x3, flow)
        d3 = torch.cat([x34t, y3, d4],dim = 1)
        d3 = self.convc1(d3)
        d3 = self.acrg3(d3)*d3
        d3 = self.mdcb3(d3)
        d31 = self.convc2(d3)
        w = self.reg3(d31)
        d3 = self.convc3(d3)
        w = self.integrate[2](w)
        flow = self.upsample_trilin(2*(self.transformer[2](flow, w) + w))

        x23t = self.transformer[1](x2, flow)
        d2 = torch.cat([x23t, y2, d3],dim = 1)
        d2 = self.convd1(d2)
        d2 = self.acrg2(d2)*d2
        d2 = self.mdcb2(d2)
        d21 = self.convd2(d2)
        w = self.reg2(d21)
        d2 = self.convd3(d2)
        w = self.integrate[1](w)
        flow = self.upsample_trilin(2*(self.transformer[1](flow, w)+w))

        x12t = self.transformer[0](x1, flow)
        d1 = torch.cat([x12t, y1, d2],dim = 1)
        d1 = self.conve1(d1)
        d1 = self.acrg1(d1)*d1
        d1 = self.mdcb1(d1)
        d1 = self.conve2(d1)
        w = self.reg1(d1)
        w = self.integrate[0](w)
        flow = self.transformer[0](flow, w)+w
        
        return flow



