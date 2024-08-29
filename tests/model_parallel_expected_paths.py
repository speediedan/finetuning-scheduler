import torch

## Expected Test Result Configuration Aliases

## example template result, providing TP weight and FSDP module states you want a test to validate
# state_key: ({p_states, fsdp_mod_states}, len(self._fts_state._curr_thawed_params))

path_tt_tp_no_fsdp = {
    0: ({'p_states': {
        'model.layers.0.feed_forward.w2.weight': {'requires_grad': False, 'is_DTensor': True},
        'model.layers.0.feed_forward.w2.bias': {'requires_grad': False, 'is_DTensor': True},
        'model.layers.1.feed_forward.w2.weight': {'requires_grad': False, 'is_DTensor': True},
        'model.layers.1.feed_forward.w2.bias': {'requires_grad': False, 'is_DTensor': True},
        'model.norm.weight': {'requires_grad': True, 'is_DTensor': True},
        'model.norm.bias': {'requires_grad': True, 'is_DTensor': True},
        'model.output.weight': {'requires_grad': True, 'is_DTensor': True}}}, 3),
    1: ({'p_states': {
        'model.layers.0.feed_forward.w2.weight': {'requires_grad': False, 'is_DTensor': True},
        'model.layers.0.feed_forward.w2.bias': {'requires_grad': False, 'is_DTensor': True},
        'model.layers.1.feed_forward.w2.weight': {'requires_grad': True, 'is_DTensor': True},
        'model.layers.1.feed_forward.w2.bias': {'requires_grad': True, 'is_DTensor': True},
        'model.norm.weight': {'requires_grad': True, 'is_DTensor': True},
        'model.norm.bias': {'requires_grad': True, 'is_DTensor': True},
        'model.output.weight': {'requires_grad': True, 'is_DTensor': True}}}, 15),
    2: ({'p_states': {
        'model.layers.0.feed_forward.w2.weight': {'requires_grad': True, 'is_DTensor': True},
        'model.layers.0.feed_forward.w2.bias': {'requires_grad': True, 'is_DTensor': True},
        'model.layers.1.feed_forward.w2.weight': {'requires_grad': True, 'is_DTensor': True},
        'model.layers.1.feed_forward.w2.bias': {'requires_grad': True, 'is_DTensor': True},
        'model.norm.weight': {'requires_grad': True, 'is_DTensor': True},
        'model.norm.bias': {'requires_grad': True, 'is_DTensor': True},
        'model.output.weight': {'requires_grad': True, 'is_DTensor': True}}}, 29),
}

path_tt_fsdp_no_tp = {
    0: (
        {'p_states': {
            'model.layers.0.feed_forward.w2.weight': {'requires_grad': False, 'is_DTensor': True},
            'model.layers.0.feed_forward.w2.bias': {'requires_grad': False, 'is_DTensor': True},
            'model.layers.1.feed_forward.w2.weight': {'requires_grad': False, 'is_DTensor': True},
            'model.layers.1.feed_forward.w2.bias': {'requires_grad': False, 'is_DTensor': True},
            'model.norm.weight': {'requires_grad': True, 'is_DTensor': True},
            'model.norm.bias': {'requires_grad': True, 'is_DTensor': True},
            'model.output.weight': {'requires_grad': True, 'is_DTensor': True}
        }},
        3),
    1: (
        {'p_states': {
            'model.layers.0.feed_forward.w2.weight': {'requires_grad': False, 'is_DTensor': True},
            'model.layers.0.feed_forward.w2.bias': {'requires_grad': False, 'is_DTensor': True},
            'model.layers.1.feed_forward.w2.weight': {'requires_grad': True, 'is_DTensor': True},
            'model.layers.1.feed_forward.w2.bias': {'requires_grad': True, 'is_DTensor': True},
            'model.norm.weight': {'requires_grad': True, 'is_DTensor': True},
            'model.norm.bias': {'requires_grad': True, 'is_DTensor': True},
            'model.output.weight': {'requires_grad': True, 'is_DTensor': True}
        }},
        15),
    2: (
        {'p_states': {
            'model.layers.0.feed_forward.w2.weight': {'requires_grad': True, 'is_DTensor': True},
            'model.layers.0.feed_forward.w2.bias': {'requires_grad': True, 'is_DTensor': True},
            'model.layers.1.feed_forward.w2.weight': {'requires_grad': True, 'is_DTensor': True},
            'model.layers.1.feed_forward.w2.bias': {'requires_grad': True, 'is_DTensor': True},
            'model.norm.weight': {'requires_grad': True, 'is_DTensor': True},
            'model.norm.bias': {'requires_grad': True, 'is_DTensor': True},
            'model.output.weight': {'requires_grad': True, 'is_DTensor': True}
        }},
        29),
}

path_tt_fsdp_tp = {
    0: ({'p_states': {
            'model.layers.0.feed_forward.w2.weight': {'requires_grad': False, 'is_DTensor': True},
            'model.layers.0.feed_forward.w2.bias': {'requires_grad': False, 'is_DTensor': True},
            'model.layers.1.feed_forward.w2.weight': {'requires_grad': False, 'is_DTensor': True},
            'model.layers.1.feed_forward.w2.bias': {'requires_grad': False, 'is_DTensor': True},
            'model.norm.weight': {'requires_grad': True, 'is_DTensor': True},
            'model.norm.bias': {'requires_grad': True, 'is_DTensor': True},
            'model.output.weight': {'requires_grad': True, 'is_DTensor': True}},
        'fsdp_mod_states': {
            'model.layers.0': {'is_fsdp_managed': True, 'is_fsdp_composed': False},
            'model.layers.1': {
                'is_fsdp_managed': True, 'is_fsdp_composed': True,
                'prec_policy_summ': (None, None, None, True),
                'param_group_summ': [
                    (None, torch.Size([200]), torch.Size([100])),
                    (None, torch.Size([200]), torch.Size([100])),
                    (None, torch.Size([200, 200]), torch.Size([100, 200])),
                    (None, torch.Size([200, 200]), torch.Size([100, 200])),
                    (None, torch.Size([200, 200]), torch.Size([100, 200])),
                    (None, torch.Size([200, 200]), torch.Size([100, 200])),
                    (None, torch.Size([200]), torch.Size([100])),
                    (None, torch.Size([200]), torch.Size([100])),
                    (None, torch.Size([800, 200]), torch.Size([400, 200])),
                    (None, torch.Size([800]), torch.Size([400])),
                    (None, torch.Size([200, 800]), torch.Size([100, 800])),
                    (None, torch.Size([200]), torch.Size([100]))]},
            'model.norm': {'is_fsdp_managed': True, 'is_fsdp_composed': True,
                           'prec_policy_summ': (None, None, None, True),
                           'param_group_summ': [
                               (None, torch.Size([200]), torch.Size([100])),
                               (None, torch.Size([200]), torch.Size([100]))]},
            'model.output': {'is_fsdp_managed': True, 'is_fsdp_composed': True,
                             'prec_policy_summ': (None, None, None, True),
                             'param_group_summ': [
                                 (None, torch.Size([33278, 200]), torch.Size([16639, 200]))]}}},
        3),
    1: ({'p_states': {
            'model.layers.0.feed_forward.w2.weight': {'requires_grad': False, 'is_DTensor': True},
            'model.layers.0.feed_forward.w2.bias': {'requires_grad': False, 'is_DTensor': True},
            'model.layers.1.feed_forward.w2.weight': {'requires_grad': True, 'is_DTensor': True},
            'model.layers.1.feed_forward.w2.bias': {'requires_grad': True, 'is_DTensor': True},
            'model.norm.weight': {'requires_grad': True, 'is_DTensor': True},
            'model.norm.bias': {'requires_grad': True, 'is_DTensor': True},
            'model.output.weight': {'requires_grad': True, 'is_DTensor': True}},
        'fsdp_mod_states': {
            'model.layers.0': {'is_fsdp_managed': True, 'is_fsdp_composed': False},
            'model.layers.1': {
                'is_fsdp_managed': True, 'is_fsdp_composed': True,
                'prec_policy_summ': (None, None, None, True),
                'param_group_summ': [
                    ('layers.1.attention_norm.weight', torch.Size([200]), torch.Size([100])),
                    ('layers.1.attention_norm.bias', torch.Size([200]), torch.Size([100])),
                    ('layers.1.attention.wq.weight', torch.Size([200, 200]), torch.Size([100, 200])),
                    ('layers.1.attention.wk.weight', torch.Size([200, 200]), torch.Size([100, 200])),
                    ('layers.1.attention.wv.weight', torch.Size([200, 200]), torch.Size([100, 200])),
                    ('layers.1.attention.wo.weight', torch.Size([200, 200]), torch.Size([100, 200])),
                    ('layers.1.ffn_norm.weight', torch.Size([200]), torch.Size([100])),
                    ('layers.1.ffn_norm.bias', torch.Size([200]), torch.Size([100])),
                    ('layers.1.feed_forward.w1.weight', torch.Size([800, 200]), torch.Size([400, 200])),
                    ('layers.1.feed_forward.w1.bias', torch.Size([800]), torch.Size([400])),
                    ('layers.1.feed_forward.w2.weight', torch.Size([200, 800]), torch.Size([100, 800])),
                    ('layers.1.feed_forward.w2.bias', torch.Size([200]), torch.Size([100]))]},
            'model.norm': {'is_fsdp_managed': True, 'is_fsdp_composed': True,
                           'prec_policy_summ': (None, None, None, True),
                           'param_group_summ': [
                               ('norm.weight', torch.Size([200]), torch.Size([100])),
                               ('norm.bias', torch.Size([200]), torch.Size([100]))]},
            'model.output': {'is_fsdp_managed': True, 'is_fsdp_composed': True,
                             'prec_policy_summ': (None, None, None, True),
                             'param_group_summ': [
                                 ('output.weight', torch.Size([33278, 200]), torch.Size([16639, 200]))]}}},
        15),
    2: (
        {
            'p_states': {
                'model.layers.0.feed_forward.w2.weight': {'requires_grad': True, 'is_DTensor': True},
                'model.layers.0.feed_forward.w2.bias': {'requires_grad': True, 'is_DTensor': True},
                'model.layers.1.feed_forward.w2.weight': {'requires_grad': True, 'is_DTensor': True},
                'model.layers.1.feed_forward.w2.bias': {'requires_grad': True, 'is_DTensor': True},
                'model.norm.weight': {'requires_grad': True, 'is_DTensor': True},
                'model.norm.bias': {'requires_grad': True, 'is_DTensor': True},
                'model.output.weight': {'requires_grad': True, 'is_DTensor': True},
            },
            'fsdp_mod_states': {
                'model.layers.0': {
                    'is_fsdp_managed': True,
                    'is_fsdp_composed': False
                },
                'model.layers.1': {
                    'is_fsdp_managed': True,
                    'is_fsdp_composed': True,
                    'prec_policy_summ': (None, None, None, True),
                    'param_group_summ': [
                        ('layers.1.attention_norm.weight', torch.Size([200]), torch.Size([100])),
                        ('layers.1.attention_norm.bias', torch.Size([200]), torch.Size([100])),
                        ('layers.1.attention.wq.weight', torch.Size([200, 200]), torch.Size([100, 200])),
                        ('layers.1.attention.wk.weight', torch.Size([200, 200]), torch.Size([100, 200])),
                        ('layers.1.attention.wv.weight', torch.Size([200, 200]), torch.Size([100, 200])),
                        ('layers.1.attention.wo.weight', torch.Size([200, 200]), torch.Size([100, 200])),
                        ('layers.1.ffn_norm.weight', torch.Size([200]), torch.Size([100])),
                        ('layers.1.ffn_norm.bias', torch.Size([200]), torch.Size([100])),
                        ('layers.1.feed_forward.w1.weight', torch.Size([800, 200]), torch.Size([400, 200])),
                        ('layers.1.feed_forward.w1.bias', torch.Size([800]), torch.Size([400])),
                        ('layers.1.feed_forward.w2.weight', torch.Size([200, 800]), torch.Size([100, 800])),
                        ('layers.1.feed_forward.w2.bias', torch.Size([200]), torch.Size([100])),
                    ],
                },
                'model.norm': {
                    'is_fsdp_managed': True,
                    'is_fsdp_composed': True,
                    'prec_policy_summ': (None, None, None, True),
                    'param_group_summ': [
                        ('norm.weight', torch.Size([200]), torch.Size([100])),
                        ('norm.bias', torch.Size([200]), torch.Size([100])),
                    ],
                },
                'model.output': {
                    'is_fsdp_managed': True,
                    'is_fsdp_composed': True,
                    'prec_policy_summ': (None, None, None, True),
                    'param_group_summ': [('output.weight', torch.Size([33278, 200]), torch.Size([16639, 200]))],
                },
            },
        },
        29,
    ),
}
