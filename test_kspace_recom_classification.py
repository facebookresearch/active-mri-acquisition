import os, sys
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from data import CreateFtTLoader
from util import util
import torchvision.utils as tvutil
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import functools
from util.util import ssim_metric
from models.fft_utils import RFFT, IFFT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.animation as animation
from util import visualizer
from models.fft_utils import create_mask
from pylab import *
rc('axes', linewidth=2)
import copy
import pickle
from sklearn.metrics import accuracy_score

rfft = RFFT()
ifft = IFFT()


def __draw_curve4(Pscore):
    # for percentile
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 3))
    # both has N * 3 matric
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10*2,10))
    plt.tight_layout()
    # assert mse.shape[1] == 3 and uncertainty.shape[1] == 3

    names = ['top10', 'median', 'bottom10']
    linestyles = ['-',':','--']
    for j, k in enumerate(Pscore.keys()):
        # import pdb; pdb.set_trace()
        # two method comparison
        mse = np.array(Pscore[k][0])
        uncertainty = np.array(Pscore[k][1])
        for i in range(3):
            key_name = f'{k.replace("_","")} {names[i]}'
            ax1.plot(mse[:,i], color=colors[j], label=key_name, linewidth=5, linestyle=linestyles[i])

        for i in range(3):
            key_name = f'{k.replace("_","")} {names[i]}'
            ax2.plot(uncertainty[:,i], color=colors[j], label=key_name, linewidth=5, linestyle=linestyles[i])

    ax1.set_ylabel('MSE', fontsize=40)
    ax1.set_xlabel('kMA+', fontsize=40)
    ax2.set_ylabel('Uncertainty', fontsize=40)
    ax2.set_xlabel('kMA+', fontsize=40)
    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    plt.tick_params(labelsize=25)
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data
def draw_curve4(Pscore):
    # for percentile
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 3))
    # both has N * 3 matric
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10*2,10))
    plt.tight_layout()
    # assert mse.shape[1] == 3 and uncertainty.shape[1] == 3

    names = ['top10', 'median', 'bottom10']
    linestyles = ['-',':','--']
    for j, k in enumerate(Pscore.keys()):
        # import pdb; pdb.set_trace()
        # two method comparison
        mse = np.array(Pscore[k][0])
        uncertainty = np.array(Pscore[k][1])
        x = range(len(mse[:,0]))
        ax1.errorbar(x, mse[:,0], color=colors[j], label=k.replace("_"," "), 
                        linewidth=5, linestyle=linestyles[i], yerr=mse[:,1])
        ax2.errorbar(x, uncertainty[:,0], color=colors[j], label=k.replace("_"," "), 
                        linewidth=5, linestyle=linestyles[i], yerr=uncertainty[:,1])

    ax1.set_ylabel('MSE', fontsize=40)
    ax1.set_xlabel('kMA+', fontsize=40)
    ax2.set_ylabel('Uncertainty', fontsize=40)
    ax2.set_xlabel('kMA+', fontsize=40)
    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    plt.tick_params(labelsize=25)
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data
def draw_curve3(uncertainty):
    global Percentage
    # uncertanity is a dict each value is a list a a[1] is var
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 5))
    fig = plt.figure(figsize=(10,10))
    plt.tick_params(labelsize=25)
    for i, (key, values) in enumerate(uncertainty.items()):
        color = 'black' if 'algorithm' in key else colors[i]
        ls = '-' if 'algorithm' in key else "--"
        data = [u[-1] for u in values] # -1 is var mean value
        key_name = key.replace('_', '+')
        xval = [float(a)/len(data) * Percentage[-1] for a in range(1, len(data)+1)]
        plt.plot(xval, data, color=color, label=key_name, linewidth=5, linestyle=ls)

    plt.legend(fontsize=25)
    plt.ylabel('Uncertainty score', fontsize=40)
    plt.xlabel('kMA+ (%)', fontsize=40)

    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data
def draw_curve2(data):
    global Percentage
    # for discriminator score
    fig = plt.figure(figsize=(20,10))
    plt.tick_params(labelsize=25)
    fig.add_subplot(111)
    # x = np.arange(data.shape[0])
    xval = [float(a)/data.shape[0] * Percentage[-1] for a in range(1, data.shape[0]+1)]
    plt.plot(xval, data[:,0], 'k', label='algorithm', linewidth=10)
    plt.fill_between(xval, data[:,0]-data[:,1], data[:,0]+data[:,1],
                    alpha=0.2, edgecolor='#FFFFFF', facecolor='#c2d5db')
    plt.ylabel('Discriminator score', fontsize=40)
    plt.xlabel('kMA+ (%)', fontsize=40)
    # plt.title('avgerage reconstruced k-space D-score', fontsize=30)
    plt.legend(fontsize=25)

    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data

def draw_curve(data):
    # for MSE and SSIM
    global Percentage
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 5))

    fig = plt.figure(figsize=(20,10))
    n = len(data)
    fig.add_subplot(121)

    for i, (key, values) in enumerate(data.items()):
        color = 'black' if 'algorithm' in key else colors[i]
        ls = '-' if 'algorithm' in key else "--"
        key_name = key.replace('_', '+')
        xval = [float(a)/values.shape[0] * Percentage[-1] for a in range(1, values.shape[0]+1)]
        plt.plot(xval, values[:,0], color=color, label=key_name, linewidth=5, linestyle=ls)
    plt.ylabel('MSE', fontsize=40)
    plt.xlabel('kMA+ (%)', fontsize=40)
    plt.legend(fontsize=25)
    plt.tick_params(labelsize=25)

    fig.add_subplot(122)
    for i, (key, values) in enumerate(data.items()):
        color = 'black' if 'algorithm' in key else colors[i]
        ls = '-' if 'algorithm' in key else "--"
        key_name = key.replace('_', '+')
        xval = [float(a)/values.shape[0] * Percentage[-1] for a in range(1, values.shape[0]+1)]
        plt.plot(xval, values[:,1], color=color, label=key_name, linewidth=5, linestyle=ls)
    plt.ylabel('SSIM', fontsize=40)
    plt.xlabel('kMA+ (%)', fontsize=40)
    plt.legend(fontsize=25)
    
    plt.tick_params(labelsize=25)
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data

def draw_curve5(data):
    # for Acc
    global Percentage
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 5))

    fig = plt.figure(figsize=(10,10))
    n = len(data)

    for i, (key, values) in enumerate(data.items()):
        color = 'black' if 'algorithm' in key else colors[i]
        ls = '-' if 'algorithm' in key else "--"
        key_name = key.replace('_', '+')
        xval = [float(a)/values.shape[0] * Percentage[-1] for a in range(1, values.shape[0]+1)]
        plt.plot(xval, values, color=color, label=key_name, linewidth=5, linestyle=ls)
    plt.ylabel('MSE', fontsize=40)
    plt.xlabel('kMA+ (%)', fontsize=40)
    plt.legend(fontsize=25)
    plt.tick_params(labelsize=25)
    
    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data

def merge_kpsace(pred, gt, mask):
    p_fft = rfft(pred[:,:1,:,:])
    g_fft = rfft(gt[:,:1,:,:])
    assert len(mask.shape) == 4
    assert mask.shape[1] == 1 and mask.shape[3] == 1
    res = g_fft * mask + p_fft * (1-mask)
    res = ifft(res)

    return res

def replace_kspace_lines(pred, gt, mask, n_line, mask_score, random=None):
    # data [B,1,128,128]
    # mask [B,1,128,1]
    # mask_score [B,128]
    # random [B,n_line]
    
    mask_new = mask.mul(1).squeeze() # copy mem
    
    if random is None:
        mask_score.masked_fill_(mask_new.byte(), 100) # do not select from real one set it to a large value
        min_ranked_score, indices = torch.sort(mask_score, 1, descending=False)
        indices = indices[:,:n_line]
    else:
        indices = random

    for i, ind in enumerate(indices):
        if mask_new[i, ind] == 1:
            import pdb; pdb.set_trace()
        mask_new[i, ind] = 1
        if conjudge_symmetric:
            mask_new[i, 128-ind] = 1

    sys.stdout.write(f'\r mask ratio: {mask_new[0].sum()}')
    sys.stdout.flush()
    mask_new = mask_new.unsqueeze(1).unsqueeze(3)
    results = merge_kpsace(pred, gt, mask_new)

    return results, mask_new

global InfoPerc
InfoPerc = []
def compute_scores(fake, real, mask):
    fake = fake[:,:1,:,:]#.clamp(-3,3)
    real = real[:,:1,:,:]
    mse = F.mse_loss(fake, real)
    ssim = util.ssim_metric(fake, real)

    get_mse_percentile(fake, real)

    # compute information percentage:
    assert mask.shape[2] == 128 
    indices = mask.squeeze().mul(1)
    h = indices.shape[1]
    pers = []
    for index in indices:
        # 2nd line to 64 are conjudge symmetric to the other part
        tmp = index.cpu().numpy()[1:64] + np.array(index.tolist()[:64:-1])
        per = np.nonzero(tmp)[0].size / 65
        pers.append(per*100//1 /100)
    InfoPerc.append(array(pers))   
    return [mse.item(), ssim.item()]

# to track the MSE per kspace recom and corresponding uncertainty 
# for top10, bottom10 and median
global MSE, Uncertainty
IndicesP = []
MSE = []
Uncertainty = []

def __get_mse_percentile(fake, real):
    mse = F.mse_loss(fake, real, reduce=False)
    mse = mse.view(mse.shape[0],-1).mean(1)
    _, indices = mse.sort(dim=0, descending=False)
    l = len(indices)
    top10 = indices[:int(l*0.1)]
    bottom10 = indices[-int(l*0.1):]
    median = indices[l//2]
    global IndicesP
    IndicesP = [top10, median, bottom10]

    _tmp = []
    for v in IndicesP:
        _tmp.append(mse[v].mean().item())
    MSE.append(_tmp)

def __get_mse_percentile_for_uncertainty(logvar):
    uncertainties = logvar.exp().view(logvar.shape[0],-1).mean(1)
    global IndicesP
    assert len(IndicesP) ==3
    
    _tmp = []
    for v in IndicesP:
        _tmp.append(uncertainties[v].mean().item())
    Uncertainty.append(_tmp)
    IndicesP = []

def get_mse_percentile(fake, real):
    mse = F.mse_loss(fake, real, reduce=False)
    _tmp = [mse.mean(), mse.std()]
    MSE.append(_tmp)

def get_mse_percentile_for_uncertainty(logvar):
    uncertainties = logvar.exp()
    Uncertainty.append([uncertainties.mean(), uncertainties.std()])
    IndicesP = []

def compute_D_score(score, mask):
    masked_pred_kspace_score = (score * (1-mask.squeeze())).reshape(-1)
    masked_pred_kspace_score = masked_pred_kspace_score[mask.squeeze().reshape(-1) == 0]
    if len(masked_pred_kspace_score) == 0:
        return [0, 0]
    else:
        return [masked_pred_kspace_score.mean(), masked_pred_kspace_score.std()]

def compute_uncertainty(logvars, ngrid=4):
    var = [logvar.exp().mean().item() for logvar in logvars]
    var_map = logvars[1][:ngrid].exp()
    # var_map = var_map.div(var_map.max())
    uncertainty_map = util.tensor2im(tvutil.make_grid(var_map, nrow=int(np.sqrt(ngrid)), padding=10), renormalize=False)
    uncertainty_map = visualizer.gray2heatmap(uncertainty_map[:,:,0])

    get_mse_percentile_for_uncertainty(logvars[-1])

    return [uncertainty_map, var[-1]]

def compute_errormap(fake, real, ngrid=4):
    error = (real - fake)[:ngrid,:1]**2
    error = error.sqrt()
    errormap = util.tensor2im(tvutil.make_grid(error, nrow=int(np.sqrt(ngrid)), padding=10, pad_value=0.6), renormalize=False)
    return errormap

def tensor2img(data, ngrid=4):
    data = util.mri_denormalize(data[:ngrid,:1].mul(1))
    imgs = util.tensor2im(tvutil.make_grid(data, nrow=int(np.sqrt(ngrid)), padding=10, pad_value=0.6), renormalize=False)
    return imgs
def compute_mask(masks, ngrid=4):
    masks = masks[:ngrid].repeat(1,1,1,opt.fineSize)
    masks = util.tensor2im(tvutil.make_grid(masks, nrow=int(np.sqrt(ngrid)), padding=10, pad_value=0.6), renormalize=False)
    return masks


def animate(images, masks, eval_scores, uncertainty, D_score, iter,
            comp_score=None, start_line=33, tot_line=128):
    # scores[0][0] is mse and [0][1] is ssim
    # images[0] is a image grid for error map
    # animate images with scires
    # uncertainty[0] image uncertainty overall value
    # comp_score has multiple comparative results in tuple
    def normalize(data):
        import pdb; pdb.set_trace()
        for i in range(len(data)):
            maxv = 0
            maxv = max(data[i].max().item(), maxv)
        for i in range(len(uncertainty)):
            data[i] = data[i] / (maxv)

    ims = []
    k = 5
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,k, figsize=(5*k,5))
    plt.tight_layout()

    # plt.title(f'# of measurement acquired: {i}')
    ax1.set_title('Error Map', fontsize=16)
    ax2.set_title('Cartesian subsampling Mask', fontsize=16)
    ax3.set_title('MSE', fontsize=16)
    ax4.set_title('Uncertainty', fontsize=16)
    ax5.set_title('Discriminator Score', fontsize=16)

    # normalize([u[0] for u in uncertainty])
    # normalize(images)

    x, y1, y2, y3, y5 = [], [], [], [], []
    tot = len(images)

    plot_property = {
       'color':'#131925', 
       'marker':'o', 
       'markeredgecolor':'#EE0000', 
       'markerfacecolor':'#EE0000', 
       'lw':5,
       'markeredgewidth': 0.5   
    }
    plot_property2 = {
       'color':'gray', 
       'lw':5,
       'linestyle': ":",
    }
    plot_property3 = {
       'color':'cyan', 
       'lw':5,
       'linestyle': "--",
    }
    global Percentage
    xval = [int(float(a)/tot * Percentage[-1]) for a in range(1, tot+1)]

    for i, n in enumerate(range(start_line, tot+start_line)):
        
        kMA = xval[i]
        ax1.axis('off')
        im = ax1.imshow(images[i], animated=True)

        ## Uncertainty map
        ax2.axis('off')
        im2 = ax2.imshow(masks[i], animated=True)

        x.append(kMA)
        y1.append(eval_scores[i][0])
        
        im3, = ax3.plot(x, y1, **plot_property, label='algorithm' if i == 0 else "")
        # text3 = ax3.text(5, 5, f'measurement {i}', color='b', fontsize=16, transform=ax3.transAxes,)
        title3 = ax3.text(0.8,0.9," kMA {}%".format(kMA), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax3.transAxes, fontsize=16)
        ax3.set_xlabel('kMA (%)', fontsize=16)
        if comp_score is not None:
            im3_2, = ax3.plot(x, [u[0] for u in comp_score[1][:i+1]], **plot_property3, label='random_reconstruction' if i == 0 else "")
            im3_3, = ax3.plot(x, [u[0] for u in comp_score[0][:i+1]], **plot_property2, label='random' if i == 0 else "")
            legend = ax3.legend(loc='lower left')
        ax3.set_xlim([0, x[-1]+1])
        
        ## SSIM score
        # y2.append(eval_scores[i][1])
        # im3, = ax3.plot(x, y2, color='gray', marker='o', markeredgecolor='r')
        # text1 = ax3.text(2,2, f'measurement {i}',animated=True, color='red', fontsize=10)
        # ax3.set_xlim([0, x[-1]+1])

        im4, = ax4.plot(x, [u[-1] for u in uncertainty[:i+1]], **plot_property)
        title4 = ax4.text(0.8,0.9," kMA {}%".format(kMA), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax4.transAxes, fontsize=16)
        ax4.set_xlabel('kMA (%)', fontsize=16)
        ax4.set_xlim([0, x[-1]+1])

        avg_D = np.array([u[0] for u in D_score[:i+1]])
        std_D = np.array([u[1] for u in D_score[:i+1]])

        im5, = ax5.plot(x, avg_D, **plot_property, label='score')
        fill5 = ax5.fill_between(x, avg_D-std_D, avg_D+std_D, label='var.', 
                    alpha=0.2, edgecolor='#FFFFFF', facecolor='#c2d5db')
        title5 = ax5.text(0.2,0.9," kMA {}%".format(kMA), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax5.transAxes, fontsize=16)
        ax5.set_xlabel('kMA (%)', fontsize=16)
        ax5.set_xlim([0, x[-1]+1])

        if comp_score is not None:
            ims.append([im, im2, im3, im3_2, im3_3, im4, im5, fill5, title3, title4, title5])
        else:
            ims.append([im, im2, im3, im4, im5, fill5, title3, title4, title5])
    plt.tight_layout()
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000)
    global opt
    global save_dir
    name = f"demo{iter}.mp4"
    save_path = os.path.join(save_dir, name)
    ani.save(save_path)
    print('\n -> video saved at', save_path)

    gif_name = f'gif_demo{iter}.gif'
    if save_gif:
        ani.save(os.path.join(save_dir, gif_name), writer='imagemagick', fps=10)
    return name, gif_name

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
def compute_accuracy(imgs, labels):
    imagnet_classifier.eval()
    # import pdb; pdb.set_trace()
    imgs = imgs[:,:1,:,:].repeat(1,3,1,1)
    imgs = imgs.add(1).div(2)
    # imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    # imgs = F.upsample(imgs, size=(224,224), mode='bilinear')
    inputs = (imgs - classifier_mean) / classifier_std
    
    with torch.no_grad():
        acc = accuracy(imagnet_classifier(inputs).cpu(), labels, topk=(1,5))
        acc = [a.item() for a in acc]

    print(f'\n img min {imgs.min().item()} max {imgs.max().item()} acc {acc}')

    # import pdb; pdb.set_trace()

    return acc

import torchvision.models as models
from torchvision import transforms

from util.train_128gray_resnet import resnet128
imagnet_classifier = resnet128.resnet50().to(0)
checkpoint_path = '/private/home/zizhao/work/checkpoint_fmri/resnset128_scratch/model_best.pth.tar'
checkpoint = torch.load(checkpoint_path)
print('checkpoint prec', checkpoint['best_prec1'])
imagnet_classifier = torch.nn.DataParallel(imagnet_classifier, device_ids=[0]).cuda()
imagnet_classifier.load_state_dict(checkpoint['state_dict'])
imagnet_classifier.eval()
classifier_mean = torch.cuda.FloatTensor([0.43, 0.43, 0.43]).view(1,3,1,1)
classifier_std = torch.cuda.FloatTensor([0.23, 0.23, 0.23]).view(1,3,1,1)

opt = None
save_dir = None
"-----------------------------------------"
## conjudge_symmetric is True. We select lines only on the top 65 lines and the bottom conjudge symmetric lines are filled automatically
# The code will adapt automatically for that 
conjudge_symmetric=True
## How many lines we observed at init. If it is euqal to 0, we use the LF 10 lines
## If = 25, we use the default 25% subsampling pattern
observed_line_n = 0
## If n_forward is None, it will calculate automatically. For debug, set it to a small value
n_forward = None
tot_iter = 1
# for displaying the xval of plot, the percentage of oberved lines
Percentage = [] 
save_metadata = False
save_gif = False
"-----------------------------------------"

if __name__ == '__main__':
    
    opt = TestOptions().parse()
    assert (opt.no_dropout)

    opt.results_dir = opt.checkpoints_dir
    # test_data_loader = CreateFtTLoader(opt, is_test=True)
    model = create_model(opt)
    model.setup(opt)
    
    import torchvision.datasets as datasets
    valdir = os.path.join('/datasets01/imagenet_resized_144px/060718/061417', 'val')
    test_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # zz
            transforms.Resize(144),
            # transforms.Resize(224),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5,0.5])
        ])),
        batch_size=opt.batchSize, shuffle=False,
        num_workers=4, pin_memory=True)

    val_count = 0
    # saved = False
    ngrid = 9 # how many image to show in each grid
    use_forward = True

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, 'test_recommend_%s' % (opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    save_dir = webpage.get_image_dir()
    # test
    total_test_n = len(test_data_loader)*opt.batchSize if opt.how_many < 0 else (opt.how_many)
    if not hasattr(model, 'sampling'): opt.n_samples = 1 # determnistic model needs no sampling
    
    if model.mri_data:
        tensor2im = functools.partial(util.tensor2im, renormalize=False)
    else:
        tensor2im = util.tensor2im

    '''parameters need to set'''
    if conjudge_symmetric:
        observed_line_n = 0 #int(init_ratio * opt.fineSize)

    if observed_line_n != 32:
        init_mask = create_mask(opt.fineSize, random_frac=False, mask_fraction=observed_line_n / opt.fineSize).to(model.device)
        observed_line_n = int(init_mask[0].sum().item())
        print('start observed line: ', observed_line_n)

    if conjudge_symmetric:
        print('>> keep conjudge symmetric')
        for jj in range(len(init_mask)):
            for j in range(1,127):
                if init_mask[jj, 0, j, 0] == 1:
                    init_mask[jj, 0, 128-j, 0] = 1

    unobsered_line_n = opt.fineSize - observed_line_n
    n_recom_each = 1
    if n_forward is None:
        if conjudge_symmetric:
            n_forward = unobsered_line_n // 2 
        else:
             unobsered_line_n // n_recom_each
    imsize = opt.fineSize
    np.random.seed(1234)

    print(f'>> start from {observed_line_n} lines and add {n_recom_each} line per forward, need {n_forward} forward')
    for i, data in enumerate(test_data_loader):
        
        print('\n Conduct iteration', i)
        # data = data[:-1]
        data[0] = data[0][:,:1,:,:]
        img_labels = data[1].mul(1)
        score_list = OrderedDict()
        visual_list = OrderedDict()
        acc_list = OrderedDict()
        uncertainty_list = OrderedDict()
        score_D = []
        visuals = {}
        
        if observed_line_n != 32:
            if init_mask.shape[0] == 1:
                init_mask = init_mask.repeat(data[0].shape[0],1,1,1)
            model.set_input_exp2(data, init_mask)
        else:
            model.set_input(data)

        '''Start point and init all scores'''
        model.test()
        mask = model.mask
        (vis_score, inv_score), pred_kspace_score = model.forward_D() # pred_kspace_score[B, 128]
        realB = model.real_B
        fakeB = model.fake_B
        init_logvars = model.logvars
        
        visual_list['groundtruth'] = tensor2img(realB)
        visual_list['input'] = tensor2img(model.real_A)
        score_list['algorithm'] = [compute_scores(fakeB, realB, mask)]
        visual_list['algorithm_errormap'] = [compute_errormap(fakeB, realB)]
        score_list['algorithm_mask'] = [ compute_mask(mask) ]

        acc_list['algorithm'] = [compute_accuracy(realB, img_labels)] # TODO change to fakeB
        
        uncertainty_list['algorithm']= [compute_uncertainty(init_logvars)]
        
        score_D.append(compute_D_score(pred_kspace_score, mask))

        PScore = {} # Pertentile score
        InfoPScore = {} # Information Percentile score
        
        bz = fakeB.shape[0]

        '''Start to recommend'''
        print('\n step 1 algorithm: ')
        old_fakeB = fakeB
        old_mask = mask
        old_mask_count = old_mask.squeeze().sum(1).mean()
        
        Percentage = [old_mask[0].sum().item() / opt.fineSize]
        for nf in range(n_forward):
            # print(f'real_A range {model.real_A.min()}-{model.real_A.max()}')
            # calcuate which line to replace and replace it
            new_fakeB, new_mask = replace_kspace_lines(old_fakeB, realB, old_mask, n_recom_each, pred_kspace_score, None)
            new_mask_count = new_mask.squeeze().sum(1).mean()
            # assert new_mask_count == old_mask_count + n_recom_each, f'{new_mask_count} != {old_mask_count} + {n_recom_each}'

            if use_forward:
                #do another forward
                model.set_input_exp2(data, new_mask)
                model.test()
                (vis_score, inv_score), pred_kspace_score = model.forward_D() # pred_kspace_score[B, 128]
                old_fakeB = model.fake_B
                score_D.append(compute_D_score(pred_kspace_score, new_mask))
            else:
                old_fakeB = new_fakeB
            old_mask = new_mask
            old_mask_count = old_mask.squeeze().sum(1).mean()

            acc_list['algorithm'].append(compute_accuracy(fakeB, img_labels))

            score_list['algorithm'].append(compute_scores(old_fakeB, realB, old_mask))

            uncertainty_list['algorithm'].append(compute_uncertainty(model.logvars))
            visual_list['algorithm_errormap'].append(compute_errormap(old_fakeB, realB))
            score_list['algorithm_mask'].append(compute_mask(old_mask))

            Percentage.append(old_mask[0].sum().item() / opt.fineSize)
            if nf < (n_forward-1) and old_mask[1].sum() == opt.fineSize:
                print ('>> reach the all lines earlier. break')
                break

        TOT_SEEN_LINE = old_mask[1].sum()
        import pdb; pdb.set_trace()
        score_D = score_D[:-1] # the last one has no meaning 
        score_D.append(score_D[-1])
        Percentage = [int(np.around(xval * 100)) for xval in Percentage]

        PScore['algorithm'] = [copy.deepcopy(MSE), copy.deepcopy(Uncertainty)]
        # InfoPScore['algorithm'] = [copy.deepcopy(InfoPerc)]
        MSE, Uncertainty, InfoPScore = [], [], []

        video_path, gif_path = animate(visual_list['algorithm_errormap'], score_list['algorithm_mask'], score_list['algorithm'], uncertainty_list['algorithm'], 
                        score_D, start_line=observed_line_n, iter=i)

        ''' Start to random select lines '''
        score_list['random'] = [compute_scores(fakeB, realB, mask)]
        acc_list['random'] = [compute_accuracy(fakeB, model.img_labels)]

        # random some lines order
        bz = mask.shape[0]
        tmp_mask = mask.mul(1)
        if conjudge_symmetric:
            tmp_mask[:,:,65:,:] = 1
        mask_zero_idx = np.nonzero(1-tmp_mask[0].squeeze())
        if not conjudge_symmetric:
            assert len(mask_zero_idx) == unobsered_line_n
        lines = [np.random.permutation(mask_zero_idx) for _ in range(bz)] # permute for each data
        cand_lines = np.stack(lines, 0).squeeze()

        old_fakeB = fakeB
        old_mask = mask
        print('\n step 2 random: ')
        
        for s in range(n_forward):
            cand_line = cand_lines[:,s*n_recom_each:(s+1)*n_recom_each]
            # calcuate which line to replace and replace it
            new_fakeB, new_mask = replace_kspace_lines(old_fakeB, realB, old_mask, n_recom_each, None, random=cand_line)

            old_fakeB = new_fakeB
            old_mask = new_mask
            score_list['random'].append(compute_scores(old_fakeB, realB, old_mask))
            acc_list['random'].append(compute_accuracy(fakeB, model.img_labels))

            if old_mask[1].sum() == opt.fineSize:
                break
        assert old_mask[1].sum() == TOT_SEEN_LINE
        # InfoPScore['random'] = [copy.deepcopy(InfoPerc)]
        MSE, Uncertainty, InfoPScore = [], [], []

        if use_forward:
            print('\n step 3 random + reconstruction: ')
            score_list['random_reconstruction'] = [compute_scores(fakeB, realB, mask)]
            uncertainty_list['random_reconstruction']= [compute_uncertainty(init_logvars)]
            acc_list['random_reconstruction'] = [compute_accuracy(fakeB, model.img_labels)]

            old_fakeB = fakeB
            old_mask = mask
                        
            for s in range(n_forward):
                cand_line = cand_lines[:,s*n_recom_each:(s+1)*n_recom_each]
                # calcuate which line to replace and replace it
                new_fakeB, new_mask = replace_kspace_lines(old_fakeB, realB, old_mask, n_recom_each, None, random=cand_line)

                #do another forward
                model.set_input_exp2(data, new_mask)
                model.test()
                (vis_score, inv_score), pred_kspace_score = model.forward_D() # pred_kspace_score[B, 128]
                old_fakeB = model.fake_B
                old_mask = new_mask
                score_list['random_reconstruction'].append(compute_scores(old_fakeB, realB, old_mask))
                uncertainty_list['random_reconstruction'].append(compute_uncertainty(model.logvars))
                acc_list['random_reconstruction'].append(compute_accuracy(fakeB, model.img_labels))
                if old_mask[1].sum() == opt.fineSize:
                    break
            assert old_mask[1].sum() == TOT_SEEN_LINE
        PScore['random_reconstruction'] = [copy.deepcopy(MSE), copy.deepcopy(Uncertainty)]

        # InfoPScore['random_reconstruction'] = [copy.deepcopy(InfoPerc)]
        MSE, Uncertainty, InfoPScore = [], [], {}
        ## animate compared results also      
        # animate(score_list['algorithm_errormap'], score_list['algorithm_mask'], score_list['algorithm'], score_list['algorithm_uncertainty'], 
        #             score_D, (score_list['random'], score_list['random_reconstruction']), start_line=observed_line_n)

        print('\n step 4 order: ')
        score_list['order'] = [compute_scores(fakeB, realB, mask)]
        acc_list['order'] = [compute_accuracy(fakeB, model.img_labels)]

        mask_zero_idx = np.nonzero(1-mask[0].squeeze())
        if not conjudge_symmetric:
            assert len(mask_zero_idx) == unobsered_line_n
        lines = [np.sort(mask_zero_idx, axis=0) for _ in range(bz)] # permute for each data
        lines = [a if np.random.rand() < 0.5 else a[::-1] for a in lines]
        cand_lines = np.stack(lines, 0).squeeze()

        old_fakeB = fakeB
        old_mask = mask
        
        for s in range(n_forward):
            cand_line = cand_lines[:,s*n_recom_each:(s+1)*n_recom_each]
            # calcuate which line to replace and replace it
            new_fakeB, new_mask = replace_kspace_lines(old_fakeB, realB, old_mask, n_recom_each, None, random=cand_line)

            old_fakeB = new_fakeB
            old_mask = new_mask
            score_list['order'].append(compute_scores(old_fakeB, realB, old_mask))
            acc_list['order'].append(compute_accuracy(fakeB, model.img_labels))
            if old_mask[1].sum() == opt.fineSize:
                break

        print('\n step 5 order + reconstruction: ')
        old_fakeB = fakeB
        old_mask = mask
        score_list['order_reconstruction'] = [compute_scores(fakeB, realB, mask)]
        uncertainty_list['order_reconstruction']= [compute_uncertainty(init_logvars)]
        acc_list['order_reconstruction'] = [compute_accuracy(fakeB, model.img_labels)]
        for s in range(n_forward):
            cand_line = cand_lines[:,s*n_recom_each:(s+1)*n_recom_each]
            # calcuate which line to replace and replace it
            new_fakeB, new_mask = replace_kspace_lines(old_fakeB, realB, old_mask, n_recom_each, None, random=cand_line)

            # do another forward
            model.set_input_exp2(data, new_mask)
            model.test()
            (vis_score, inv_score), pred_kspace_score = model.forward_D() # pred_kspace_score[B, 128]
            old_fakeB = model.fake_B
            old_mask = new_mask
            score_list['order_reconstruction'].append(compute_scores(old_fakeB, realB, old_mask))
            uncertainty_list['order_reconstruction'].append(compute_uncertainty(model.logvars))
            acc_list['order_reconstruction'].append(compute_accuracy(fakeB, model.img_labels))
            if old_mask[1].sum() == opt.fineSize:
                break
        assert old_mask[1].sum() == TOT_SEEN_LINE
        # InfoPScore['order_reconstruction'] = [copy.deepcopy(InfoPerc)]
        MSE, Uncertainty, InfoPScore = [], [], []


        "record all thing"
        score = OrderedDict()

        score['random_C'] = np.array(score_list['random'])
        if use_forward:
            score['random_C_R'] = np.array(score_list['random_reconstruction'])
        score['order_C'] = np.array(score_list['order'])
        score['order_C_R'] = np.array(score_list['order_reconstruction'])
        score['algorithm_C_R'] = np.array(score_list['algorithm'])
        
        acc = OrderedDict()
        score_D = np.array(score_D)
        for k, v in acc_list.items():
            acc[k] = np.array(v)

        # curves of uncertainty
        score_uncetainty = OrderedDict()
        score_uncetainty['random_C_R'] = uncertainty_list['random_reconstruction']
        score_uncetainty['order_C_R'] = uncertainty_list['order_reconstruction']     
        score_uncetainty['algorithm_C_R'] = uncertainty_list['algorithm']   

        ## draw curves 
        visuals['performance'] = draw_curve(score)
        visuals['D-score'] = draw_curve2(score_D)
        visuals['uncertainty'] = draw_curve3(score_uncetainty)
        visuals['acc'] = draw_curve5(acc)

        visuals['demo'] = video_path
        visuals['groundtruth'] = visual_list['groundtruth'] 
        visuals['input'] = visual_list['input'] 
        
        if save_gif:
            visuals['gif'] = gif_path

        # save data
        if save_metadata:
            MetaData = {}
            MetaData['score'] = score
            MetaData['score_D'] = score_D
            MetaData['data'] = data
            meta_name = f'metadata{i}.pickle'
            with open(os.path.join(save_dir, meta_name),'wb') as f:
                pickle.dump(MetaData, f)
                print('save at', os.path.join(save_dir, meta_name))

            visuals['metadata'] = meta_name

        ## (deprecated) draw percentile mse vs uncertainty
        ## visuals['model_percentile'] = draw_curve4(PScore)

        ## save figures and everthing
        save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        webpage.save()

        val_count += bz

        # if val_count >= opt.how_many and opt.how_many > 0:
        #     break
        if i >= tot_iter: break

    
    # save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # webpage.save()   
    #  
    # import pdb; pdb.set_trace()
    # if i >= 10: break

        # import pdb; pdb.set_trace()

        # visuals = model.get_current_visuals()
        # val_count += ngrid
        # # Only save some. Don't explode the disk
        # if val_count < 256:
        #     for k, v in visuals.items():
        #         v = v[:ngrid,:1,:,:]
        #         if model.mri_data:
        #             v = util.mri_denormalize(v)
        #         visuals[k] = tensor2im(tvutil.make_grid(v, nrow=int(np.sqrt(ngrid))))
            
        #     save_images(webpage, visuals, 'test_iter{}'.format(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        # elif not saved:
        #     webpage.save()
        #     saved = True
            
        # if val_count >= opt.how_many and opt.how_many > 0:
        #     break
