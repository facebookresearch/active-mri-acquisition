# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch.nn.parameter import Parameter
#
#
# class AUTOPARAM(nn.Module):
#     def __init__(self, length=3, fixed_ws=None):
#         super(AUTOPARAM, self).__init__()
#         self.use_fixed_w = fixed_ws is not None
#         if fixed_ws is None:
#             print('[AUTOPARAM] -> use trainable weight')
#             self.weight = Parameter(torch.FloatTensor(length).fill_(1))
#         else:
#             print('[AUTOPARAM] -> use fixed weight',fixed_ws)
#             self.register_buffer('weight', torch.FloatTensor(fixed_ws))
#
#     def forward(self):
#         if self.use_fixed_w:
#             weight = self.weight
#         else:
#             weight = F.softmax(self.weight) * self.weight.shape[0] # sum to be one
#
#         return weight
