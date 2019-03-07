import math
import numpy as np
import logging
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from torch.nn import functional as F
import pathlib
import sys
import textwrap
import pdb
import torch
from torch.utils.data import DataLoader
from . import subsample, dicom_dataset

try:
    Sampler = torch.utils.data.Sampler
except:
    Sampler = torch.utils.data.sampler.Sampler

class FixedSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def example_predictions(model, args,
    fname="grid.png", fold="val", grid_size=32, runinfo=None):
    """
    Saves a visualization of the ground truth and predictions on the first minibatch
    of the validation set.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_func = subsample.Mask(reuse_mask=True) # Fix for the grid
    dataset = dicom_dataset.Slice(
        mask_func=mask_func,
        args=args,
        which=fold,
    )
    # Fix the exact images we get rather than sampling
    sample_range = list(range(10, 330, 10))
    sample_range[:min(grid_size, len(sample_range))]
    sampler = FixedSampler(sample_range)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        sampler=sampler
    )

    grid_predictions = None
    grid_images = None
    idx = 0
    model.eval()
    if args.log_during_eval:
        logging.info(f"Starting image_grid loop, batch_size={args.batch_size}")
        sys.stdout.flush()
    #pdb.set_trace()

    with torch.no_grad():
        for masked_kspace, image, mask, metadata in data_loader:
            masked_kspace = masked_kspace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            if args.apex:
                image = image.half()
                masked_kspace = masked_kspace.half()
                mask = mask.half()

            prediction = model(masked_kspace, mask)

            if grid_predictions is None:
                grid_predictions = torch.zeros(grid_size, prediction.shape[1],
                    prediction.shape[2], prediction.shape[3]).cuda()
                grid_images = torch.zeros_like(grid_predictions)
                if args.log_during_eval:
                    logging.info("Created grid objects")
                    sys.stdout.flush()
                #pdb.set_trace()

            for j in range(prediction.shape[0]):
                grid_predictions[idx, ...] = prediction.data[j, ...].float()
                grid_images[idx, ...] = image.data[j, ...].float()
                idx += 1
                if idx >= grid_size:
                    break
            if idx >= grid_size:
                break

    save(grid_predictions, grid_images, fname=fname, runinfo=runinfo)



def save(predicted, ground, fname="grid.png", runinfo=None):
    """
        Pass in a single minibatch
    """
    predicted = predicted.detach()
    nimages = predicted.shape[0]
    nw = predicted.shape[3]
    if nw != predicted.shape[2]:
        raise Exception("Currently only square images supported")
    nbh = nw+2 # n with border pixels
    caption_h = 20
    nbw = (nw+2)*3
    grid_max_width = 1170
    grid_blocks_wide = int(grid_max_width / (nbh*3))
    grid_blocks_heigh = int(math.ceil(nimages / grid_blocks_wide))
    offset_x = 200 # header region

    def loc(i):
        x = offset_x + (nbh+caption_h)*int(i / grid_blocks_wide)
        y = nbw*int(i % grid_blocks_wide)
        return x,y

    ar = np.zeros([offset_x+grid_blocks_heigh*(nbh+caption_h),  grid_max_width, 3])
    for i in range(nimages):
        x,y = loc(i)
        gi = ground[i, 0, :, :]
        shift = torch.min(gi)
        scale = torch.max(gi-shift)
        # Blue border around ground truth
        ar[x:(x+nw+2), y:(y+nw+2), 2] = 0.8
        ar[(x+1):(x+nw+1), (y+1):(y+nw+1), :] = (ground[i, 0, :, :][..., None]-shift)/scale
        y += nbh
        ar[(x+1):(x+nw+1), (y+1):(y+nw+1), :] = (predicted[i, 0, :, :][..., None]-shift)/scale
        y += nbh
        ar[(x+1):(x+nw+1), (y+1):(y+nw+1), :] = 0.5 + 4*((ground-predicted)[i, 0, :, :][..., None])/scale

    ar *= 255
    ar = np.clip(ar,0,255)

    path = pathlib.Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)

    #pdb.set_trace()
    img_pil = Image.fromarray(ar.astype('uint8'), mode='RGB')

    ### Header part
    header_txt = str(runinfo["args"])
    text_width = 160
    header_txt = textwrap.fill(header_txt, width=text_width)
    if len(header_txt) > text_width*9:
        header_txt = header_txt[:text_width*10]

    try:
        header_txt += f"\n Epoch {runinfo['epoch']}"
        header_txt += f"\n Latest test loss {runinfo['test_losses'][-1]}"
    except:
        pass

    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default() #truetype("sans-serif.ttf", 8)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((5, 5), header_txt, (255,255,255), font=font)

    # Draw captions
    for i in range(nimages):
        x, y = loc(i)
        loss = F.mse_loss(ground[i, 0, :, :], predicted[i, 0, :, :]).item()
        draw.text((y+nbh+5, x+nbh+3), f"{loss:1.5f}", (255,255,255), font=font)
        #pdb.set_trace()

    img_pil.save(fname, format="PNG")
    logging.info(f"Saved image grid to {fname}")
    sys.stdout.flush()
