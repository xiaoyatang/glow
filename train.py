from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
import os
import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow
from convert_RGB_to_H_or_E_prostate import convert_RGB_to_H_or_E
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=2, type=int, help="batch size") #16
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations") #200000
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks") #default:4
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", default=True, help="use affine coupling instead of additive"
) #refer to section3.1 in paper. Actnorm layer performs an affine transformation of the activations using a scale and bias per channel, similar to BN.
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=400, type=int, help="image size")#224
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("--path", default="PATH",metavar="PATH", type=str, help="Path to image directory")


def sample_data(path, batch_size, image_size=400):
    transform = transforms.Compose(
        [
            # transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path,transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel #c in equation 2 of paper, c= -Mloga, M: dim of x; a: related to discretization level of data
    loss = loss + logdet + log_p #loss = -(log(p(z))+sum(log|w|)+c)

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )
def MSEloss(tarImgH,tarImgE,z1_outs):
    z1_H = z1_outs[:,:6,:,:]
    z1_E = z1_outs[:,6:,:,:]
    z1_H = torch.reshape(z1_H, (z1_H.size(0), -1))
    z1_E = torch.reshape(z1_E, (z1_E.size(0), -1))
    z1_H = z1_H[:,:tarImgH.size(2)*tarImgH.size(1)]
    z1_E = z1_E[:,:tarImgH.size(2)*tarImgH.size(1)]
    tarImgH = torch.reshape(tarImgH, (tarImgH.size(0), -1))
    tarImgE = torch.reshape(tarImgE, (tarImgE.size(0), -1))
    # z1_H,z1_E = convert_RGB_to_H_or_E(z1_outs[:2],'./Test_set_256x256_H_my_code',\
    #         './Test_set_256x256_E_my_code',args.batch,args.img_size)
    mse_loss = torch.nn.functional.mse_loss(z1_H,tarImgH)+torch.nn.functional.mse_loss(z1_E,tarImgE)
    return mse_loss,z1_H,z1_E

def train(args, model, optimizer):
    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2.0 ** args.n_bits

    z_sample = [] #to generate new images from random sample
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)#（channels,input_size,input_size)(6, 112, 112), (12, 56, 56), (24, 28, 28), (96, 14, 14)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp #20,6,112,112...
        z_sample.append(z_new.to(device)) #list of 4
        
    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset) #B images,B labels
            image = image.to(device)  #[B,3,224,224]
            oriImg = 255*image.clone().detach() #float32 [B,3,224,224] scale to original pixel value
            tarImgH,tarImgE = convert_RGB_to_H_or_E(oriImg,'./Test_set_256x256_H_my_code',\
            './Test_set_256x256_E_my_code',image.size(0),image.size(-1))
            norm_min = image.min()
            norm_max = image.max()
            image = (image-norm_min)/(norm_max-norm_min) * 255

            if args.n_bits < 8:  #noise
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5 #n_bins = 2.0 ** n_bits

            if i == 0:
                with torch.no_grad():  #temporarily sets all of the requires_grad flags to false
                    log_p, logdet, z_outs,z1_outs = model.module(
                        image + torch.rand_like(image) / n_bins
                    )  #CALL FORWARD log_p_sum[batch,1], logdet[batch,1], z_outs([batch,6,200size/2,200size/2],[batch,16,100size/4,100size/4],
                    #[batch,24,50size/8,50size/8],[batch,96,25,25])

                    continue

            else:
                log_p, logdet, z_outs,z1_outs= model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins) #loss: sum loss; log_p: log p(z); log_det: sum(log(det|w|))
            # with torch.no_grad():
            #     generated_samples = model_single.reverse(z_sample)
                # utils.save_image(
                #         generated_samples.cpu().data,
                #         f"./sample/{str(i + 1).zfill(6)}.png",
                #         normalize=True,
                #         nrow=10,
                #         range=(-0.5, 0.5),
                #     )
            # mseLoss,z1_H,z1_E = MSEloss(tarImgH,tarImgE,generated_samples)
            mseLoss,z1_H,z1_E = MSEloss(tarImgH,tarImgE,z1_outs[0])
            loss += 0.00001* mseLoss
            model.zero_grad()
            loss.backward() #do bp on sumLoss
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; mseLoss:{mseLoss.item():.5f};logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )
            # pbar.set_description(
            #     f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            # )

            if i % 100 == 0: #save image per 100 iters 问题是只保存前200个epochs
                with torch.no_grad():
                    z1_H = torch.reshape(z1_H, (z1_H.size(0), 1, 400, 400))
                    z1_E = torch.reshape(z1_E, (z1_E.size(0), 1, 400, 400))
                    tarImgH = tarImgH.unsqueeze(1)
                    tarImgE = tarImgE.unsqueeze(1)
                    to_pil_image = transforms.ToPILImage(mode='L')
                    for i in range(z1_H.size(0)):
                        pil_image_Htar = to_pil_image(tarImgH[i])
                        filename = f'image_{i}.png'
                        tar_H_path = os.path.join('./Train_set_400x400_H_my_code', filename)
                        pil_image_Htar.save(tar_H_path)
                        pil_image_Etar = to_pil_image(tarImgE[i])
                        filename = f'image_{i}.png'
                        tar_E_path = os.path.join('./Train_set_400x400_E_my_code', filename)
                        pil_image_Etar.save(tar_E_path)
                        #save H&E 
                        pil_image_H = to_pil_image(z1_H[i])
                        file_H_path = os.path.join('./Test_set_400X400_H_my_code/class2', filename)
                        pil_image_H.save(file_H_path)
                        pil_image_E = to_pil_image(z1_E[i])
                        filename = f'imageE_{i}.png'
                        file_E_path = os.path.join('./Test_set_400X400_H_my_code/class2', filename)
                        pil_image_E.save(file_E_path)
                        #save generated views
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        # generated_samples.cpu().data,
                        f"./sample/alpha1eminus4/{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )
                    #random samples torch.Size([20, 3, 224, 224]),f"sample/{str(i + 1).zfill(6)}.png",

            if i % 10000 == 0: #save model per 10000 iters
                torch.save(
                    model.state_dict(), f"checkpoint/model_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer.state_dict(), f"checkpoint/optim_{str(i + 1).zfill(6)}.pt"
                )


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)