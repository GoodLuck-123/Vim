# Complete config for CNN baseline depth estimation

_base_ = [
    '../_base_/models/cnn_baseline.py',
    '../_base_/datasets/nyu_depth_v2.py',
    '../_base_/schedules/schedule_60k.py',
    '../_base_/default_runtime.py'
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
        loss_decode=dict(
            type='SILogLoss', variance_focus=0.85, loss_weight=1.0)),
)

# Reduce batch size and iterations for quick test
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
)

total_iters = 1000
find_unused_parameters = True

# Disable validation to avoid eval issues during sanity check
evaluation = dict(by_epoch=False, interval=999999)  # Disable validation
