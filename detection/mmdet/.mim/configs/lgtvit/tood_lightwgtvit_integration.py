
default_scope = 'mmdet'
custom_imports = dict(
    imports=['mmdet.models.backbones.multiscalelgtformer'],
    allow_failed_imports=False)

model = dict(
    type='TOOD',
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
        # embed_dim=96, # base
        embed_dim=72,  # tiny, small
        # depths=[2, 2, 6, 3],  # tiny
        depths=[3, 5, 8, 5],  # small
        # depths=[4, 8, 21, 5], # 4, 8: local stages, 21, 4: global stages
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
        dilate_attention=['dilate', 'dilate', 'dcn', 'dcn'],
        downsamples=[True, True, True, True],
        cpe_per_satge=False, cpe_per_block=True,
        offset_scale=1.0,
        dw_kernel_size=3,
        center_feature_scale=False,
        remove_center=False,
        output_bias=True,
        without_pointwise=False,
        out_indices=(0, 1, 2, 3),
        task='DET',
        # init_cfg=dict(type='Pretrained', checkpoint='/root/pretrain/imagenet1k/checkpoint.pth')
        init_cfg=None
    ),
    neck=dict(
        type='FPN',
        in_channels=[144, 288, 576, 1152],
        # in_channels=[192, 384, 768, 1536],  # neck input channels == output channel of each stage in backbone
        out_channels=256,  # output channels of each layer of fpn network
        num_outs=5),  # output scales
    bbox_head=dict(
        type='TOODHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        num_dcn=2),

    train_cfg=dict(
        initial_epoch=8,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),

    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)

)
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(eps=1e-08, lr=1e-4, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=11),
    dict(
        T_max=25,  # iters/epochs from crest to trough or trough to crest
        begin=11,
        by_epoch=True,
        convert_to_iter_based=True,
        end=36,
        eta_min=1e-6,
        type='CosineAnnealingLR'),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
# schedule_1x.py
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')