import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.harmonization_dataset import HarmonizationValidation
import torch.utils.data as data

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
        default="/data1/liguanlin/Datasets/iHarmony/Hday2night/composite_images_test/"

    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/harmonization-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    opt = parser.parse_args()

    #整理数据
    #masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    #images = [x.replace("_mask.png", ".png") for x in masks]
    #print(f"Found {len(masks)} inputs.")
    #整理数据这里我们直接使用创建Dataset的方式
    dataset = HarmonizationValidation()
    dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    #加噪模型
    config = OmegaConf.load("models/ldm/harmonization/harmonization-inference.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/harmonization/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    #实例化采样器
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    #对所有的输入照片，进行循环采样
    os.makedirs(opt.outdir, exist_ok=True)


    with torch.no_grad():
        with model.ema_scope():
            for batch in dataloader:
                for k in batch:
                    if torch.is_tensor(batch[k]):
                        batch[k] = batch[k].to(device=device)
                outpath = os.path.join(opt.outdir, batch["path"][0])

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["cond_image"]) #使用image的背景，前景全0作为encoder的输入
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:]) #将mask进行缩放
                #c = torch.cat((c, cc), dim=1) #将mask和masked_image concate起来作为conditioning

                shape = (c.shape[1],)+c.shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 mask=cc,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)#对扩散结果进行解码

                #下面3句话中clamp的作用是将像素值控制在0到1之间
                image = torch.clamp((batch["gt_image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                harmonized = (1-mask)*image+mask*predicted_image #从预测得到的照片中获取前景，背景仍然使用原始图片
                harmonized = harmonized.cpu().numpy().transpose(0,2,3,1)[0]*255 #乘以255得到可视化的RGB图像。
                Image.fromarray(harmonized.astype(np.uint8)).save(outpath) #保存得到的RGB图像
