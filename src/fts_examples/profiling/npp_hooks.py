import torch

# accessed in global scope to track non-parameter packed bytes (npp) as a simple proxy (ceiling) for activation memory
_npp_bytes = 0

def _hook_npp_pre_forward(module, *args, **kwargs):
    mem = module.mem_info_handle()
    global _npp_bytes
    module.npp_pre_forward = _npp_bytes
    module.rss_pre_forward = mem.rss
    return None

def _hook_npp_post_forward(module, *args, **kwargs):
    global _npp_bytes
    module.npp_post_forward = _npp_bytes
    module.npp_diff = module.npp_post_forward - module.npp_pre_forward
    mem = module.mem_info_handle()
    module.rss_post_forward = mem.rss
    rss_diff = module.rss_post_forward - module.rss_pre_forward
    module.rss_diff = rss_diff + (module.rss_diff if hasattr(module, "rss_diff") else 0)
    return None

def _reset_memory_hooks_state(model, reset_attrs: list[str]):
    global _npp_bytes
    _npp_bytes = 0
    for module in model.modules():
        for attr in reset_attrs:
            setattr(module, attr, 0)

def _npp_hook(x):
    global _npp_bytes
    if not isinstance(x, torch.nn.Parameter):
        _npp_bytes += x.nbytes
    return x
