import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange
from ops.dcn.deform_conv import ModulatedDeformConv


class Net(nn.Module):
    def __init__(self, in_nc, angular_out, factor):
        """
        Args:
            in_nc: num of input channels.
            angular_out: output angular resolution.
            factor: spatial upsampling scale.
        """
        super(Net, self).__init__()
        nf, nf_s, nb, base_ks, block = 64, 64, 3, 3, 8
        """
            nf, nf_s: channel number of feature
            nb: num of down/up sample.
            deform_ks: size of the deformable kernel.
            block: number of FSI block
        """
        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = 3
        self.size_dk = 3 ** 2
        self.an2 = angular_out*angular_out

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc*in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                    )
                )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                    )
                )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )

        self.offset_mask = nn.Conv2d(
            nf, in_nc*in_nc*3*self.size_dk*angular_out*angular_out, 1, padding=0
            )
        
        self.deform_conv = ModulatedDeformConv(
            in_nc*in_nc, nf_s, self.deform_ks, padding=self.deform_ks//2, deformable_groups=in_nc*in_nc
            )
        self.SpatialRefine = CascadedBlocks(block, nf_s, angular_out)
        self.UpSample = Upsample(nf_s, factor)

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks        
        mv_input = LFsplit(inputs, self.in_nc)
        # feature extraction (with downsampling)
        out_lst = [self.in_conv(mv_input.contiguous())]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
                )
        # Adaptive pixel aggregation
        off_msk_all = self.offset_mask(self.out_conv(out))
        off_msk_list = torch.split(off_msk_all, in_nc*in_nc*3*n_off_msk, dim=1)
        out_fea = []
        for i in range(len(off_msk_list)):
            off_msk = off_msk_list[i]
            off = off_msk[:, :in_nc*in_nc*2*n_off_msk, ...]
            msk = torch.sigmoid(
                off_msk[:, in_nc*in_nc*2*n_off_msk:, ...]
                )
            # perform deformable convolutional fusion
            fused_feat = F.relu(
                self.deform_conv(mv_input, off, msk), 
                inplace=True
                )
            out_fea.append(fused_feat)
        multi_fea = torch.stack(out_fea, 1) ##b,n,c,h,w

        b,n,c,h,w = multi_fea.shape
        feats = self.SpatialRefine(multi_fea)
        out = FormOutput(self.UpSample(feats))
                
        return out

class Upsample(nn.Module):
    def __init__(self, channel, factor):
        super(Upsample, self).__init__()
        self.upsp = nn.Sequential(
            nn.Conv2d(channel, channel * factor * factor, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(factor),
            nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b*n, -1, h, w)
        out = self.upsp(x)
        _, _, H, W = out.shape
        out = out.contiguous().view(b, n, -1, H, W)
        return out

class CascadedBlocks(nn.Module):
    def __init__(self, n_blocks, channel, angRes):
        super(CascadedBlocks, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(FSI(channel, angRes))
        self.body = nn.Sequential(*body)
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer = x        
        for i in range(self.n_blocks):
            buffer = self.body[i](buffer)
        buffer = self.conv(buffer.contiguous().view(b*n, c, h, w))
        return buffer.contiguous().view(b, n, c, h, w) + x

class FSI(nn.Module):
    '''
    Feature Separation and Interaction
    '''
    def __init__(self, ch, angRes):
        super(FSI, self).__init__()
                
        self.relu = nn.ReLU(inplace=True)
        S_ch, A_ch, Eh_ch, Ev_ch  = ch, ch//4, ch//2, ch//2
        self.spaconv  = SpatialConv(ch, S_ch)
        self.angconv  = AngularConv(ch, angRes, A_ch)
        self.epiconv = EPiConv(ch, angRes, Eh_ch)

        self.spaconv_f  = SpatialConv(S_ch+A_ch, S_ch, kernel=3)
        self.angconv_f  = AngularConv(S_ch+A_ch, angRes, A_ch, kernel=3)
        self.epihconv_f = EPiConv(Eh_ch+Eh_ch, angRes, Eh_ch, kernel=3)
        self.epivconv_f = EPiConv(Eh_ch+Eh_ch, angRes, Ev_ch, kernel=3)
        
        self.fuse = nn.Sequential(
                        nn.Conv2d(in_channels = S_ch+A_ch+Eh_ch+Ev_ch, out_channels = ch, kernel_size = 1, stride = 1, padding = 0, dilation=1))
    
    def forward(self,x):
        b, n, c, h, w = x.shape
        an = int(math.sqrt(n))
        s_out = self.spaconv(x)
        a_out = self.angconv(x)
        epi_in = x.contiguous().view(b, an, an, c, h, w)
        epih_out = self.epiconv(epi_in)
        multi_xT = rearrange(epi_in, 'b u v c h w -> b v u c w h')
        epiv_out = rearrange(self.epiconv(multi_xT), 'b (v u) c w h -> b (u v) c h w', u=an, v=an)

        s_inter = self.spaconv_f(torch.cat((s_out, a_out), 2))+s_out
        a_inter = self.angconv_f(torch.cat((a_out, s_out), 2))+a_out
        epi_f_in = torch.cat((epih_out, epiv_out), 2)
        epi_f_in = epi_f_in.contiguous().view(b, an, an, -1, h, w)
        eh_inter = self.epihconv_f(epi_f_in)+epih_out
        multi_xT_f = rearrange(epi_f_in, 'b u v c h w -> b v u c w h')
        ev_inter = rearrange(self.epivconv_f(multi_xT_f), 'b (v u) c w h -> b (u v) c h w', u=an, v=an)+epiv_out
       
        out = torch.cat((s_inter, a_inter, eh_inter, ev_inter), 2)

        out = self.fuse(out.contiguous().view(b*n, -1, h, w))
        return out.contiguous().view(b,n,c,h,w) + x

class SpatialConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel=3):
        super(SpatialConv, self).__init__()
        self.spaconv_s = nn.Sequential(
                            nn.Conv2d(ch_in, ch_out, kernel_size=kernel, stride=1, padding=kernel//2, bias=True),
                            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.spachannel = ch_out

    def forward(self,fm):
        b, n, c, h, w = fm.shape
        buffer = fm.view(b*n, c, h, w)  
        out = self.spaconv_s(buffer)
        out = out.contiguous().view(b,n,self.spachannel,h,w)
        return out

class AngularConv(nn.Module):
    def __init__(self, ch, angRes, AngChannel, kernel=3):
        super(AngularConv, self).__init__()
        self.angconv = nn.Sequential(
                            nn.Conv2d(ch, AngChannel, kernel_size=kernel, stride=1, padding=kernel//2, bias=True),
                            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.an = angRes

    def forward(self,fm):
        b, n, c, h, w = fm.shape
        an = self.an
        a_in = fm.view(b,n,c,h*w)
        a_in = torch.transpose(a_in,1,3)
        a_in = a_in.contiguous().view(b*h*w,c,an,an)
        a_out = self.angconv(a_in)
        a_out = a_out.contiguous().view(b, h*w,-1,an*an)
        a_out = torch.transpose(a_out,1,3)
        out = a_out.contiguous().view(b, n, -1, h, w)
        return out

class EPiConv(nn.Module):
    def __init__(self, ch, angRes, EPIChannel, kernel=3):
        super(EPiConv, self).__init__()
        self.epi_ch = EPIChannel
        self.epiconv = nn.Sequential(
                        nn.Conv2d(ch, EPIChannel, kernel_size=kernel, stride=1, padding=kernel//2, bias=True),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True))
        #self.an = angRes

    def forward(self,fm):
        b, u, v, c, h, w = fm.shape
        epih_in  = fm.permute(0, 1, 4, 3, 2, 5).contiguous().view(b*u*h,c,v,w)
        epih_out = self.epiconv(epih_in)
        out = epih_out.view(b, u, h, self.epi_ch, v, w).permute(0, 1, 4, 3, 2, 5).contiguous().view(b, u*v,self.epi_ch,h,w)
        return out
def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st.squeeze(2)


def FormOutput(x_sv):
    b, n, c, h, w = x_sv.shape
    angRes = int(math.sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net = Net(2, 5, 2).cuda()
    from thop import profile
    input = torch.randn(1, 1, 64, 64).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
