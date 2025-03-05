custom_imports = dict(
    imports=['mmdet.models.backbones.multiscalelgtformer'],
    allow_failed_imports=False)

dataset_type = 'CocoDataset'
data_root = '/root/data/coco2017/'
find_unused_parameters = True
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    # '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # Image normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        # Image padding parameters
        pad_mask=True,  # In instance segmentation, the mask needs to be padded
        pad_size_divisor=1),  # Padding the image to multiples of 32
    backbone=dict(
        type='MultiScaleLgtFormer',
        # img_size=(800, 1333),
        img_size=(1024, 1024),
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=72,
        depths=[2, 2, 6, 2],
        groups=[4, 8, 18, 36],
        num_heads=[3, 6, 12, 24],
        kernel_size=3,
        dilation=[1, 2, 3],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.1,
        norm_layer='layer_norm',
        merging_way='conv3_2',
        patch_way='overlaping',
        dilate_attention=[True, True, False, False],
        downsamples=[True, True, True, False],
        cpe_per_satge=False, cpe_per_block=True,
        offset_scale=1.0,
        dw_kernel_size=3,
        center_feature_scale=False,
        remove_center=False,
        output_bias=True,
        without_pointwise=False,
        init_cfg=dict(type='Pretrained', checkpoint='/root/pretrain/imagenet1k/checkpoint.pth')
        # init_cfg=dict(type='Pretrained', checkpoint='/root/output/imagenet/checkpoint.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[144, 288, 576],  # neck input channels == output channel of each stage in backbone
        out_channels=256,  # output channels of each layer of fpn network
        num_outs=5),  # output scales
)

# override coco_detection.py
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),

    dict(
        type='Resize',
        # scale=(1333, 1333),  # Resize to ensure the longer side is 1333 while keeping the aspect ratio
        scale=(1024, 1024),
        keep_ratio=True
    ),
    # dict(
    #     type='Pad',
    #     size_divisor=1,  # Disable automatic padding
    #     pad_val=0,  # Padding value, 0 for black
    # ),
    dict(pad_val=dict(img=(
        0,
        0,
        0,
    )), size=(
        1024,
        1024,
    ), type='Pad'),

    # dict(
    #     type='Resize',
    #     # scale=(1333, 800),
    #     scale=(1333, 800),
    #     keep_ratio=False  # Keep_ratio is False to force the resizing to (1333, 800)
    # ),
    # dict(
    #     type='Pad',
    #     size=(1333, 800),  # Pad the image to (1333, 800)
    #     pad_val=dict(img=(0, 0, 0)),  # The padding value (0 for black padding)
    # ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
val_pipeline = [
    dict(type='LoadImageFromFile'),

    dict(
        type='Resize',
        # scale=(1333, 1333),  # Resize to ensure the longer side is 1333 while keeping the aspect ratio
        scale=(1024, 1024),
        keep_ratio=True
    ),
    # dict(
    #     type='Pad',
    #     size_divisor=1,  # Disable automatic padding
    #     pad_val=0,  # Padding value, 0 for black
    # ),
    dict(pad_val=dict(img=(
        0,
        0,
        0,
    )), size=(
        1024,
        1024,
    ), type='Pad'),
    # dict(
    #     type='Resize',
    #     scale=(1333, 800),
    #     keep_ratio=False
    # ),
    # dict(
    #     type='Pad',
    #     size=(1333, 800),  # Pad the image to (1333, 800)
    #     pad_val=dict(img=(0, 0, 0)),  # The padding value (0 for black padding)
    # ),

    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = val_pipeline

backend_args = None
train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001) # 0.02 -> 0.0001
# )
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=1e-4,
#         weight_decay=0.05,
#         eps=1e-8
#     ),
#     clip_grad=None,
# )

# optim_wrapper = dict(
#     type='OptimWrapper',
#     # optimizer
#     optimizer=dict(
#         type='AdamW',
#         lr=1e-4,
#         weight_decay=0.05,
#         eps=1e-8,
#         betas=(0.9, 0.999)),
#     clip_grad=None,
#     # Parameter-level learning rate and weight decay settings
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone': dict(lr_mult=0.1, decay_mult=1.0),
#         },
#         norm_decay_mult=0.0),

    # gradient clipping
    # clip_grad=dict(max_norm=0.01, norm_type=2)
# )
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(eps=1e-08, lr=1e-4, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')

# Used to dynamically adjust the parameters of the optimizer during training,
# the most common being learning rate scheduling. It can change hyper parameters
# such as learning rate according to predefined rules to promote model convergence
# param_scheduler = [
    # dict(
        # type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=12,
    #     by_epoch=True,
    #     milestones=[8, 11],
    #     gamma=0.1)
#     dict(
#         type='CosineAnnealingLR',
#         T_max=8,
#         eta_min=1e-8,
#         begin=0,
#         end=8,
#         by_epoch=True,
#         convert_to_iter_based=True),
# ]
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=30),
    dict(
        T_max=70,   # iters/epochs from crest to trough or trough to crest
        begin=30,
        by_epoch=True,
        convert_to_iter_based=True,
        end=100,
        eta_min=1e-6,
        type='CosineAnnealingLR'),
    # dict(
    #     T_max=12,
    #     begin=12,
    #     by_epoch=True,
    #     convert_to_iter_based=True,
    #     end=24,
    #     eta_min=1e-6,
    #     type='CosineAnnealingLR'),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
