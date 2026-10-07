import torch
print('available', torch.cuda.is_available(), 'count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    print('calculation', (torch.ones(1, device='cuda:0') + 1).item())
