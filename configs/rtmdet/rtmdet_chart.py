_base_ = './rtmdet_l_8xb32-300e_coco.py'

load_from = './checkpoint/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'

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
        _delete_=True,
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

interval = 1
max_epochs = 50

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=0,
    val_interval=interval)
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=10,
        save_best='auto'
    ))