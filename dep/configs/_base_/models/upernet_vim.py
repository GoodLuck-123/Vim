# --------------------------------------------------------
# Depth Estimation Model Config for Vision Mamba
# Adapted from semantic segmentation baseline for monocular depth prediction
# --------------------------------------------------------
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='VisionMambaSeg',
        in_chans=3,
        patch_size=16,
        embed_dim=384,
        depth=12,
    ),
    decode_head=dict(
        type='DepthHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        min_depth=1e-3,
        max_depth=10.0,
        loss_decode=dict(
            type='SILogLoss', variance_focus=0.85, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

find_unused_parameters = True
