_base_ = './yolov3_d53_8xb8-ms-416-273e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=1))

data_root = 'data/chart/'
metainfo = {
    'classes': ('data',),
    'palette': [
        (220, 20, 60), 
    ]
}
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='questions_multistep_descriptive_train_coco.json',
        data_prefix=dict(img='images/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='questions_multistep_descriptive_test_coco.json',
        data_prefix=dict(img='images/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'questions_multistep_descriptive_test_coco.json')
test_evaluator = val_evaluator

# Training configuration for small dataset
interval = 5  # Validate every 5 epochs instead of every epoch
max_epochs = 100  # Increased epochs for small dataset

# Optimizer configuration (add this section)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', 
        lr=0.001,  # Reduced learning rate for fine-tuning
        momentum=0.9, 
        weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2),
    # paramwise_cfg=dict(
    #     norm_decay_mult=0,  # Don't decay batch norm parameters
    #     bias_decay_mult=0,  # Don't decay bias parameters
    # )
)

# Learning rate scheduler for fine-tuning
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000),  # Warmup for first 1000 iterations
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30, 40],  # Reduce LR at these epochs
        gamma=0.1)
]


train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=0,
    val_interval=interval)
# Hooks configuration
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=interval,
        max_keep_ckpts=5,  # Keep fewer checkpoints to save space
        save_best='auto'),  # Save best model based on validation metric
    logger=dict(
        type='LoggerHook',
        interval=10),  # Log every 10 iterations
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'))
