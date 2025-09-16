_base_ = "./grounding_dino_swin-b_pretrain_mixeddata.py"

data_root = 'data/chart/'
class_name = ('data', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

# model = dict(bbox_head=dict(num_classes=num_classes))


test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]


val_dataloader = dict(
    dataset=dict(
        # metainfo=metainfo,
        type='ChartRefDataset',
        data_root=data_root,
        ann_file='questions_multistep_descriptive_test_vg_odvg.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        return_classes=True,
        backend_args=None))


test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='ChartRefExpMetric',
    ann_file=data_root + 'questions_multistep_descriptive_test_vg_odvg.json',
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))
test_evaluator = val_evaluator