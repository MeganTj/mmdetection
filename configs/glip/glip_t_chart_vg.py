_base_ = './glip_atss_swin-t_fpn_dyhead_pretrain_obj365-goldg-cc3m-sub.py'

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_mmdet-c24ce662.pth'  # noqa
lang_model_name = 'bert-base-uncased'
data_root = 'data/chart/'
class_name = ('data', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

# model = dict(bbox_head=dict(num_classes=num_classes))
# model = dict(
#     type='GLIP',
#     data_preprocessor=dict(
#         type='DetDataPreprocessor',
#         mean=[103.53, 116.28, 123.675],
#         std=[57.375, 57.12, 58.395],
#         bgr_to_rgb=False,
#         pad_size_divisor=32),
#     backbone=dict(
#         type='SwinTransformer',
#         embed_dims=96,
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 24],
#         window_size=7,
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.2,
#         patch_norm=True,
#         out_indices=(1, 2, 3),
#         with_cp=False,
#         convert_weights=False),
#     neck=dict(
#         type='FPN',
#         in_channels=[192, 384, 768],
#         out_channels=256,
#         start_level=0,
#         relu_before_extra_convs=True,
#         add_extra_convs='on_output',
#         num_outs=5),
#     bbox_head=dict(
#         type='ATSSVLFusionHead',
#         lang_model_name=lang_model_name,
#         num_classes=256,
#         in_channels=256,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             ratios=[1.0],
#             octave_base_scale=8,
#             scales_per_octave=1,
#             strides=[8, 16, 32, 64, 128],
#             center_offset=0.5),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoderForGLIP',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[0.1, 0.1, 0.2, 0.2]),
#     ),
#     language_model=dict(type='BertModel', name=lang_model_name),
#     train_cfg=dict(
#         assigner=dict(type='ATSSAssigner', topk=9),
#         allowed_border=-1,
#         pos_weight=-1,
#         debug=False),
#     test_cfg=dict(
#         nms_pre=1000,
#         min_bbox_size=0,
#         score_thr=0.05,
#         nms=dict(type='nms', iou_threshold=0.6),
#         max_per_img=100))

# test_pipeline = [
#     dict(
#         type='LoadImageFromFile',
#         backend_args=_base_.backend_args,
#         imdecode_backend='pillow'),
#     dict(
#         type='FixScaleResize',
#         scale=(800, 1333),
#         keep_ratio=True,
#         backend='pillow'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'text', 'custom_entities', 'tokens_positive', 'dataset_mode'))
# ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
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
                   'tokens_positive', 'dataset_mode'))
]

# train_dataloader = dict(
#     batch_size=1,
#     dataset=dict(
#         _delete_=True,
#         type='CocoDataset',
#         data_root=data_root,
#         metainfo=metainfo,
#         return_classes=True,
#         pipeline=train_pipeline,
#         filter_cfg=dict(filter_empty_gt=False, min_size=32),
#         ann_file='questions_multistep_descriptive_train_full_coco.json',
#         data_prefix=dict(img='images/'))
# )

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
    _delete_=True,
    type='ODVGDataset',
    data_root='data/chart/',
    ann_file='questions_multistep_descriptive_train_vg_odvg.json',
    # label_map_file=None,
    data_prefix=dict(img='images/'),
    # actual_dataset_mode='VG',
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None))


val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        # metainfo=metainfo,
        type='ChartRefDataset',
        data_root=data_root,
        ann_file='questions_multistep_descriptive_test_vg_odvg.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline, # IMPORTANT 
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
max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'language_model': dict(lr_mult=0.),
            'backbone': dict(lr_mult=0.0),
        }),
    clip_grad=None)