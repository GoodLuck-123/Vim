# Improved CNN baseline config for stable training
# Applied improvements:
# 1. Reduced learning rate (1e-4 → 5e-5)
# 2. Extended warmup (1500 → 5000 iters)
# 3. Higher warmup ratio (1e-6 → 0.1)
# 4. Added gradient clipping for stability
# 5. Extended training to verify convergence

_base_ = [
    '../_base_/models/cnn_baseline.py',
    '../_base_/datasets/nyu_depth_v2.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        type='CNNBaseline',
        in_chans=3,
        embed_dim=64,
        depths=(2, 2, 6, 2),
    ),
    decode_head=dict(
        type='DepthHead',
        in_channels=[64, 128, 256, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        min_depth=1e-3,
        max_depth=10.0,
        loss_decode=dict(
            type='SILogLoss',
            variance_focus=0.85,
            loss_weight=1.0
        )),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# Extended training: test full 60k iterations
total_iters = 60000

# Data config
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
)

# CRITICAL: Improved optimizer for numerical stability
optimizer = dict(
    type='AdamW',
    lr=5e-5,  # REDUCED: 1e-4 → 5e-5 (50% reduction)
    betas=(0.9, 0.999),
    weight_decay=0.05,
)

# CRITICAL: Improved learning rate schedule
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=5000,  # EXTENDED: 1500 → 5000 (3.3x longer)
    warmup_ratio=0.1,   # INCREASED: 1e-6 → 0.1 (linear ramp)
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# CRITICAL: Added gradient clipping for stability
optimizer_config = dict(
    type='DistOptimizerHook',
    grad_clip=dict(max_norm=1.0),  # NEW: Prevent gradient explosion
    coalesce=True,
    bucket_size_mb=-1,
)

# Runner config
runner = dict(type='IterBasedRunner')

# Checkpoint config - save more frequent checkpoints for debugging
checkpoint_config = dict(
    by_epoch=False,
    interval=5000,  # Save every 5k iters
    max_keep_ckpts=3,
)

# Log config
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ],
)

# Evaluation config - validate every 10k iters
evaluation = dict(
    by_epoch=False,
    interval=10000,
    metric='mde',
)

# Runtime settings
find_unused_parameters = True

# Disable fp16 for stability
fp16 = None
