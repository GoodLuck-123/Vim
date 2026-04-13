# --------------------------------------------------------
# Depth estimation config for Vision Mamba Small (24 layers)
# Dataset: NYU Depth v2
# Input size: 512x512
# Training iterations: 60k
# --------------------------------------------------------

_base_ = [
    '../../_base_/models/upernet_vim.py',
    '../../_base_/datasets/nyu_depth_v2.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_60k.py'
]

crop_size = (512, 512)

model = dict(
    backbone=dict(
        type='VisionMambaSeg',
        img_size=512,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=24,
        out_indices=[5, 11, 17, 23],
        pretrained=None,
        rms_norm=True,
        residual_in_fp32=False,
        fused_add_norm=True,
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type='v2',
        final_pool_type='all',
        if_divide_out=True,
        if_cls_token=False,
    ),
    decode_head=dict(
        in_channels=[384, 384, 384, 384],
        channels=512,
    ),
    test_cfg=dict(mode='whole')
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.92)
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

# By default, models are trained on 4 GPUs with 8 images per GPU
data = dict(samples_per_gpu=8, workers_per_gpu=16)

runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)
