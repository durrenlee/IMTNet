# custom_imports = dict(
#     imports=['mmdet.models.backbones.multiscalelgtformer'],
#     allow_failed_imports=False)

dataset_type = 'InsulatorDefectImgsDataset'
data_root = '/root/workspace/idid-coco/'
find_unused_parameters = True
_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    # '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    # '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

metainfo = dict(
    # classes=('No issues', 'Broken', 'Flashover damage', 'notbroken-notflashed'),
    classes=('No issues', 'Broken', 'Flashover damage'),
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
             (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
             (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
             (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
             (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
             (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
             (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
             (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
             (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
             (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
             (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
             (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
             (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
             (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
             (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
             (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
             (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
             (246, 0, 122), (191, 162, 208)])

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
        img_size=(800, 1333),
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        # embed_dim=96,
        embed_dim=72,  # tiny
        depths=[2, 2, 6, 2],  # tiny
        # depths=[4, 8, 21, 4], # 4, 8: local stages, 21, 4: global stages
        # depths=[4, 8, 10, 3],
        # depths=[4, 9, 22, 5],
        # depths=[6, 10, 24, 5],
        groups=[4, 8, 18, 36], # for dim:72
        # groups=[4, 8, 24, 48], # for dim:96
        # groups=[4, 12, 24, 48],  # for dim:96
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
        init_cfg=dict(type='Pretrained', checkpoint='/root/workspace/pretrain/checkpoint.pth')
        # init_cfg=None
    ),
    neck=dict(
        type='FPN',
        in_channels=[144, 288, 576],
        # in_channels=[144, 288, 576, 1152],
        # in_channels=[192, 384, 768, 1536],  # neck input channels == output channel of each stage in backbone
        out_channels=256,  # output channels of each layer of fpn network
        num_outs=5),  # output scales
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    # roi_head=dict(
    #     bbox_head=[
    #         dict(
    #             type='ConvFCBBoxHead',
    #             num_shared_convs=4,
    #             num_shared_fcs=1,
    #             in_channels=256,
    #             conv_out_channels=256,
    #             fc_out_channels=1024,
    #             roi_feat_size=7,
    #             num_classes=3,
    #             bbox_coder=dict(
    #                 type='DeltaXYWHBBoxCoder',
    #                 target_means=[0., 0., 0., 0.],
    #                 target_stds=[0.1, 0.1, 0.2, 0.2]),
    #             reg_class_agnostic=False,
    #             reg_decoded_bbox=True,
    #             norm_cfg=dict(type='SyncBN', requires_grad=True),
    #             loss_cls=dict(
    #                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #             loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
    #         dict(
    #             type='ConvFCBBoxHead',
    #             num_shared_convs=4,
    #             num_shared_fcs=1,
    #             in_channels=256,
    #             conv_out_channels=256,
    #             fc_out_channels=1024,
    #             roi_feat_size=7,
    #             num_classes=3,
    #             bbox_coder=dict(
    #                 type='DeltaXYWHBBoxCoder',
    #                 target_means=[0., 0., 0., 0.],
    #                 target_stds=[0.05, 0.05, 0.1, 0.1]),
    #             reg_class_agnostic=False,
    #             reg_decoded_bbox=True,
    #             norm_cfg=dict(type='SyncBN', requires_grad=True),
    #             loss_cls=dict(
    #                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #             loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
    #         dict(
    #             type='ConvFCBBoxHead',
    #             num_shared_convs=4,
    #             num_shared_fcs=1,
    #             in_channels=256,
    #             conv_out_channels=256,
    #             fc_out_channels=1024,
    #             roi_feat_size=7,
    #             num_classes=3,
    #             bbox_coder=dict(
    #                 type='DeltaXYWHBBoxCoder',
    #                 target_means=[0., 0., 0., 0.],
    #                 target_stds=[0.033, 0.033, 0.067, 0.067]),
    #             reg_class_agnostic=False,
    #             reg_decoded_bbox=True,
    #             norm_cfg=dict(type='SyncBN', requires_grad=True),
    #             loss_cls=dict(
    #                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #             loss_bbox=dict(type='GIoULoss', loss_weight=10.0))]),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,  # class number change
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
)

# override coco_detection.py
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),

    # standard data augmentation
    dict(
        type='Resize',
        # type='RandomResize',
        scale=(1333, 800),
        # scale=(224, 224),
        # scale=[(800, 800), (1333, 1333)],
        keep_ratio=False
    ),
    # dict(
    #     type='Pad',
    #     size_divisor=1,  # Disable automatic padding
    #     pad_val=0,  # Padding value, 0 for black
    # ),
    # dict(pad_val=dict(img=(
    #     0,
    #     0,
    #     0,
    # )), size=(
    #     224,
    #     224,
    # ), type='Pad'),
    # dict(type='MixUp', ratio_range=(0.5, 1.5)),
    # dict(type='Mosaic', prob=0.1),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    # dict(type='RandomShift', prob=0.1),
    # dict(type='CutOut', n_holes=(1, 3), cutout_shape=None, cutout_ratio=(0.2, 0.4), fill_in=(0, 0, 0)),
    # dict(type='Pad', size_divisor=32),

    # Albumentations

    # pack
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),

    dict(
        type='Resize',
        # scale=(800, 800),  # Resize to ensure the longer side is 1333 while keeping the aspect ratio
        # scale=(224, 224),
        scale=(1333, 800),
        keep_ratio=False
    ),
    # dict(
    #     type='Pad',
    #     size_divisor=1,  # Disable automatic padding
    #     pad_val=0,  # Padding value, 0 for black
    # ),
    # dict(pad_val=dict(img=(
    #     0,
    #     0,
    #     0,
    # )), size=(
    #     800,
    #     1333,
    # ), type='Pad'),

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
# train_dataset = dict(
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='labels_v1.2_train_coco.json',
#         data_prefix=dict(img='train/'),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         metainfo=metainfo,
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
#         ],
#         backend_args=backend_args),
#     pipeline=train_pipeline)
# val_dataset = dict(
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='labels_v1.2_val_coco.json',
#         data_prefix=dict(img='val/'),
#         test_mode=True,
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             # dict(type='LoadAnnotations'),
#         ],
#         backend_args=backend_args),
#     pipeline=val_pipeline)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='labels_v1.2_train_coco.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=metainfo,
        pipeline=train_pipeline,
        backend_args=backend_args
        ))
    # dataset=train_dataset)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='labels_v1.2_val_coco.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))
    # dataset=val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'labels_v1.2_val_coco.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


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
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=30),
    dict(
        T_max=70,  # iters/epochs from crest to trough or trough to crest
        begin=30,
        by_epoch=True,
        convert_to_iter_based=True,
        end=100,
        eta_min=1e-6,
        type='CosineAnnealingLR'),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
