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
    batch_size=4,
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

interval = 5  # Validate every 5 epochs instead of every epoch
max_epochs = 100  # Increased epochs for small dataset

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=0,
    val_interval=interval)
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=20
    ))