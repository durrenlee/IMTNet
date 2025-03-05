from mmdet.apis import init_detector, inference_detector
# 配置文件和预训练权重的路径
config_file = 'configs/lgtvit/mask_rcnn_lightwgtvit_t_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/mask_rcnn_lightwgtvit_t_fpn_1x_coco/epoch_33.pth'
device = 'cpu'  # 或者 'cpu'

# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)

# 测试图片和输出文件的路径
img = 'test.jpg'  # 或者视频文件的路径

# 进行检测
result = inference_detector(model, img)

# model.show_result(img, result, out_file='test_out.jpg')# save image with result