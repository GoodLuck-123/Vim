# CNN Baseline model config

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
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
        norm_cfg=norm_cfg,
        align_corners=False,
        min_depth=1e-3,
        max_depth=10.0,
        loss_decode=dict(
            type='SILogLoss', variance_focus=0.85, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

find_unused_parameters = True
