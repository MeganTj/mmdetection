_base_ = './faster-rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1)))

data_root = 'data/chart/'
metainfo = {
    'classes': ('data',),
    'palette': [
        (220, 20, 60), 
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='questions_multistep_descriptive_train_full_coco.json',
        data_prefix=dict(img='images/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='questions_multistep_descriptive_test_full_coco.json',
        data_prefix=dict(img='images/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'questions_multistep_descriptive_test_full_coco.json')
test_evaluator = val_evaluator

interval = 1  # Validate every epoch
max_epochs = 100  # Increased epochs for small dataset

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=0,
    val_interval=interval)
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=5,
        save_best='auto'
    ))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[70, 90],  # Reduce LR at these epochs
        gamma=0.1)
]