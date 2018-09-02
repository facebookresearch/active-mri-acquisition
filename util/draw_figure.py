import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import torchvision.utils as tvutil
from util import visualizer
from util import util

def draw_histogram(data, pdfsavepath):
    # data [3,K]
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 5))
    fig = plt.figure(figsize=(10,8))
    plt.tick_params(labelsize=15)
    th=0.2
    data[data>0.6] = 0.0
    idx = data[0,:] > th # take the high uncertainty area of stage 1

    # the histogram of the data
    for o, x in enumerate(data):
        label = 'stage'+str(o+1) if o > 0 else 'stage'+str(o+1)+f' (area > {th})'
        color = colors[o] if o < 2 else 'red'
        n, bins, patches = plt.hist(x[idx], 50, normed=1, facecolor=color, 
                            alpha=0.75, label=label)

    plt.legend(fontsize=20)
    # plt.ylabel('', fontsize=20)
    plt.xlabel('Uncertainty score', fontsize=25)

    plt.savefig(pdfsavepath, bbox_inches='tight')

    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data

def draw_curve(xval, yval, ylabel, label='', color='black', ls='--', pdfsavepath=None):
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(xval, yval, color=color, label=label, linewidth=5, linestyle=ls)

    plt.ylabel(ylabel, fontsize=40)
    plt.xlabel('kMA+ (%)', fontsize=40)
    if label != '':
        plt.legend(fontsize=25)
    plt.tick_params(labelsize=25)
    
    if pdfsavepath is not None:
        plt.savefig(pdfsavepath, bbox_inches='tight')

    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data

def draw_curve_msevar(mask_ratio, mse, ylabel, label='', color='black', ls='--', pdfsavepath=None):
    
    # mask_ratio k vector
    # mse [N, k]
    mask_ratios = np.repeat(np.array(mask_ratio)[np.newaxis,...], mse.shape[0], axis=0)
    
    
    # plt.plot(xval, yval, color=color, label=label, linewidth=5, linestyle=ls)
    min_mse = mse.min()
    max_mse = np.median(mse)
    mse_range = np.linspace(min_mse, max_mse, 6)[1:-1]

    mse_range = [0.002, 0.006, 0.01, 0.05, 0.1]
    Nplot = len(mse_range)
    fig, axs = plt.subplots(Nplot,1, figsize=(5*Nplot,10))
    yvals = []
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 5))
    labels = []
    for i, mse_t in enumerate(range(Nplot)):
        if i == 0:
            cond = (mse - mse_t) < 0
            labels.append('< %.3f' % mse_t)
        elif i == len(mse_range):
            cond = (mse - mse_t[i-1]) >= 0
            labels.append('> %.3f' % mse_t)
        else:
            labels.append('%.3f-%.3f' % (mse_range[i-1], mse_t))
            cond = np.logical_and((mse - mse_t) <= 0, (mse - mse_range[i-1]) > 0)

        yvals.append(mask_ratios[np.nonzero(cond)])
        axs[i].scatter(yvals[-1], [1 for _ in range(len(yvals[-1]))], lw=2, color=colors[i], alpha=0.5)
        axs[i].set_ylabel(f'MSE {labels[-1]}', fontsize=16)
        axs[i].set_xlabel('# of k-spsace lines observed', fontsize=16)
        # axs[i].get_yaxis().set_visible(False)
        axs[i].tick_params(labelsize=16)


    # ## box plot more intuitive than scatter
    # bplot1 = plt.boxplot(yvals,
    #                      vert=True,  # vertical box alignment
    #                      patch_artist=True,  # fill with color
    #                      labels=labels)  # will be used to label x-ticks

    # colors = ['pink', 'lightblue', 'lightgreen']
    # for patch, color in zip(bplot1['boxes'], colors):
    #     patch.set_facecolor(color)
    # plt.xlabel('MSE', fontsize=25)
    # plt.ylabel('# of k-spsace lines observed', fontsize=25)
    # plt.tick_params(labelsize=16)
    
    if pdfsavepath is not None:
        plt.savefig(pdfsavepath, bbox_inches='tight')

    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data

def compute_mse_uncertainty_cor(data, label='', how_many_to_display=1, pdfsavepath=None):
    # data [T, Nmask, 4] or [T, Nmask, 2] 
    # data[:,0] is mse 
    T, N, K = data.shape
    if K == 2:
        keys = ['uncertainty']
    else:
        # our full method
        keys = ['uncertainty_stage1', 'uncertainty_stage2', 'uncertainty_stage3']

    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 5))
    fig = plt.figure(figsize=(10,8))
    plt.tick_params(labelsize=15)
    
    for i, k in enumerate(keys[:1]):   
        c = 'navy' #colors[i+2] # if i < len(keys)-1 else 'black' 
        xval = data[:,:,i+1].mean(0).numpy()
        yval = data[:,:,0].mean(0).numpy()
        
        for s in range(data.shape[1]):
            plt.scatter(data[:,s,i+1].numpy(), data[:,s,:0].numpy(), lw=5, color=c, alpha=0.5)

        # plt.plot(xval, yval, label=label, lw=5, color='black')

    plt.xlim(0, 0.125)
    if label != '':
        plt.legend(fontsize=20)
    plt.xlabel('Uncertainty score', fontsize=20)
    plt.ylabel('MSE', fontsize=25)
    plt.title('Correlation (per sample)', fontsize=25)

    if pdfsavepath is not None:
        plt.savefig(pdfsavepath, bbox_inches='tight')

    fig.tight_layout()
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data