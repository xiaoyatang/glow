import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1)) #【1,12,1,1】
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8)) #nn.Parameter would be optimized but not with buffer params.
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1) #Actnorm palys like a BN
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1) #fill the tensor by 1

        log_abs = logabs(self.scale) #s 1,12,1,1

        logdet = height * width * torch.sum(log_abs) #h*w*sum(log|s|) |det W| of actnorm

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel) #first randomly initialize W then decomposite it by LU
        q, _ = la.qr(weight) #compute the ar factorization of a matrix(因式分解)
        #Any real square matrix A may be decomposed as A=QR,q is orthogonal qT = q-1,and r is an upper triangular matrix.
        w_p, w_l, w_u = la.lu(q.astype(np.float32)) #compute pivoted LU decomposition of a matrix(主元分解) A=PLU
        #P is a permutation matrix, L lower triangular with unit diagonal, and U upper triangular.
        w_s = np.diag(w_u) # for easily calculate det|w| as log|s| when W=PL(w_U+diag(w_s)) ,w_s is a vector
        w_u = np.triu(w_u, 1) #Diagonal above main Diagonal elements, copy U as an upper tri matrix but with 0 on the diagonal
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T #lower unit triangualr matrix with main diagonal of 0

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s) #The given NumPy array is not writable, and PyTorch does not support non-writable tensors. 
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p) # not optimize?
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s)) #sign of the the w_s
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0])) #12x12 identity matrix
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape #Batch,Channel,224,224

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)#log determinant of actnorm -- h*w*sum(log|s|)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s))) #upper tri U+ s as diagonal.
        ) #weights that can be decompisted to PL(U+diag(s)

        return weight.unsqueeze(2).unsqueeze(3) #12,12,1,1

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)) #[W-1]12,12,1,1


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_() #ZERO init is better in paper
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True), #inplace=True means that it will modify the input directly, without allocating any additional output.
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05) #initialization for first conv layer
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05) #initialization for second conv layer
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)  #2,6,112,112;2,6,112,112,split the input to two equal part

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1) #NN in paper ,self.net(in_a) 2,12,112,112
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2) #2,6,112,112
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1) #2x1

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel) #1x1 convolution with the weight matrix can be decomposited by LU

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input) #input2,12,112,112, out of actnorm 2,12,112,112, log(det|W|) is a number
        out, det1 = self.invconv(out) #out of invconv 2,12,112,112, log(det|Wconv|) is a number
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2) #[b,3,112,2,112,2]reshape the input without copying memory
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4) #[b, 3, 2, 2, 112, 112]
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2) #2,12,112,112

        logdet = 0

        for flow in self.flows:
            out, det = flow(out) ##[2,12,112,112],[2,1]
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)  #will split a z_i after every block.
            mean, log_sd = self.prior(out).chunk(2, 1) #go into the prior layer conv(6,12,k=3,s=1) after 32 flows in a block
            log_p = gaussian_log_p(z_new, mean, log_sd) #2,6,112,112
            log_p = log_p.view(b_size, -1).sum(1) #batch,1

        else:
            zero = torch.zeros_like(out) #2,96,14,14 in the last block
            mean, log_sd = self.prior(zero).chunk(2, 1) #go into the prior layer conv(96,192,k=3,s=1) after 32x4 flows in 4 blocks->2,192,14,14
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1) #B3prior:20,48,28,28;

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z #B4prior:20,96,14,14;

        for flow in self.flows[::-1]:
            input = flow.reverse(input) #B4flows:20,96,14,14;B3flows:20,48,28,28;

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        ) #reverseB4:20,24,28,28;reverseB3:20,12,56,56;reverseB3:20,6,112,112;

        return unsqueezed 


class Glow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input #[2,3,224,224]
        z_outs = []
        z1_outs = []
        for block in self.blocks:
            out, det, log_p, z_new = block(out) #[2,6,112,112];[1];[1,2];[2,6,112,112] out and z_new are the same at last layer
            if out.size(1)==6:
                out1 = torch.cat((out,z_new),dim=1)
                z1_outs.append(out1) #use part the output after block 1 as ~x
            z_outs.append(z_new)
            logdet = logdet + det #sum(log(det|W|)) of 32x4 flows in 4 blocks

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs, z1_outs   #z_outs([2,6,112,112],[2,12,56,56],[2,24,28,28],[2,96,14,14])

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input