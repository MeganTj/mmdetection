_base_ = './my_co_dino_5scale_swin_l_16xb1_16e_o365tococo.py'

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

interval = 1  # Validate every 5 epochs instead of every epoch
max_epochs = 30  # Increased epochs for small dataset

train_cfg = dict(
    val_interval=interval,
    max_epochs=max_epochs)
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=10,  # Keep fewer checkpoints to save space
        save_best='auto'
    ))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[18],
        gamma=0.1)
]