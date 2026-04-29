norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='VisionMambaSeg',
        in_chans=3,
        patch_size=16,
        embed_dim=192,
        depth=24,
        img_size=512,
        out_indices=[5, 11, 17, 23],
        pretrained='work_dirs/vim_t_midclstok_76p1acc.pth',
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type='v2',
        final_pool_type='all',
        if_divide_out=True,
        if_cls_token=False),
    decode_head=dict(
        type='DepthHead',
        in_channels=[192, 192, 192, 192],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=192,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='GN', requires_grad=True, num_groups=32),
        align_corners=False,
        min_depth=0.001,
        max_depth=10.0,
        loss_decode=dict(type='SILogLoss', loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
find_unused_parameters = True
dataset_type = 'NYUDepthV2Dataset'
data_root = '../data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthAnnotation', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='NYUDepthV2Dataset',
        data_root='../data',
        img_dir='nyu2_train',
        ann_dir='nyu2_train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadDepthAnnotation', reduce_zero_label=False),
            dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='NYUDepthV2Dataset',
        data_root='../data',
        img_dir='nyu2_test',
        ann_dir='nyu2_test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='NYUDepthV2Dataset',
        data_root='../data',
        img_dir='nyu2_test',
        ann_dir='nyu2_test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.02)
optimizer_config = dict(type='OptimizerHook', grad_clip=dict(max_norm=5.0))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=3)
evaluation = dict(
    interval=1000,
    metric='depth',
    save_best='AbsRel',
    rule='less',
    greater_keys=[],
    less_keys=['AbsRel'])
fp16 = None
work_dir = 'work_dirs/depth_vim_tiny_24_512_60k'
gpu_ids = range(0, 1)
