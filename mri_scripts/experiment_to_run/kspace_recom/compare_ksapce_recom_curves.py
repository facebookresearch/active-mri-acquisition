import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot  import imsave
plt.switch_backend('agg')

path = '/private/home/zizhao/work/checkpoint_fmri/mri_session_v2/'
methods = {
   'Design 1 (full method)': 'knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm',
   'Design 2 (two heads)': 'knee_energypasnetplus_w111logvar_0.1gan_nogradctx_pxlm_claaux', #'knee_energypasnetplusclaaux_w113logvar_0.5gan_nogradctx_pxlm'
   'Design 3 (group conv)': 'knee_energypasnetplus_w111logvar_0.1gan_nogradctx_pxlm_groupD',
}

# methods = {
#    '0.1': 'knee_energypasnetplus_w111logvar_0.1gan_gradctx_pxlm',
#    '0.01': 'knee_energypasnetplus_w111logvar_0.01gan_gradctx_pxlm',
#    '0.05': 'knee_energypasnetplus_w111logvar_0.05gan_gradctx_pxlm',
#    '0.2': 'knee_energypasnetplus_w111logvar_0.2gan_gradctx_pxlm'
# }
file_path = 'test_recommend_latest/images//metadata0.pickle'

fig = plt.figure(figsize=(10,10))
colors = matplotlib.cm.rainbow(np.linspace(0, 1, 10))

pdfsavepath = 'figures/acquisition planning ablation threshold.pdf'

tot = -1
for i, (method_name, method) in enumerate(methods.items()):
    cur_file_path = os.path.join(path, method, file_path)
    print('load', cur_file_path)
    MetaData = pickle.load(open(cur_file_path, 'rb'))
    Percentage = MetaData['Percentage']
    if i == 0 and False:
        for s, (key, values) in enumerate(MetaData['score'].items()):
            color = 'black' if 'ours' in key else colors[s]
            ls = '-' if 'ours' in key else "--"
            alpha = 1 if 'ours' in key else 0.5
            key_name = method_name if 'ours' in key else key.replace('_', '+')
            xval = [float(a)/values.shape[0] * Percentage[-1] for a in range(1, values.shape[0]+1)]
            plt.plot(xval[:tot], values[:tot,0], color=color, label=key_name, linewidth=5, linestyle=ls, alpha=alpha)
    else:
        values = MetaData['score']['ours_C_R'] if 'ours_C_R' in MetaData['score'].keys() else  MetaData['score']['algorithm_C_R']
        ls = ':' 
        color = colors[i]
        if i == 0: color = 'black'
        key_name = method_name.replace('_', '+')
        xval = [float(a)/values.shape[0] * Percentage[-1] for a in range(1, values.shape[0]+1)]
        plt.plot(xval[:tot], values[:tot,0], color=color, label=key_name, linewidth=5, linestyle=ls)


plt.ylabel('MSE', fontsize=40)
plt.xlabel('kMA+ (%)', fontsize=40)
plt.legend(fontsize=25)
plt.tick_params(labelsize=25)
fig.tight_layout()
fig.canvas.draw()

# Now we can save it to a numpy array.
data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

imsave(pdfsavepath.replace('pdf','png'), data)
# plt.savefig(pdfsavepath, bbox_inches='tight')
