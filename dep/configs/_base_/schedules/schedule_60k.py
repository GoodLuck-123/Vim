# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=4)
# evaluation: use AbsRel (absolute relative error) as metric, lower is better
evaluation = dict(interval=1000, metric='depth', save_best='AbsRel', rule='less', greater_keys=[], less_keys=['AbsRel'])

