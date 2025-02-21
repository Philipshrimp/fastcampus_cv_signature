import mmcv

from mmdet3d.apis import init_model, inference_mono_3d_detector
from mmdet3d.visualization import Det3DLocalVisualizer

if __name__=="__main__":
    init_args = {}
    call_args = {}

    init_args['model'] = "configs/imvoxelnet_2xb4_sunrgbd-3d-10class.py"
    init_args['weights'] = "ckpt/imvoxelnet_4x2_sunrgbd-3d-10class_20220809_184416-29ca7d2e.pth"
    init_args['device'] = "cuda:0"

    call_args["inputs"] = dict(img="data/sunrgbd/000017.jpg", infos="data/sunrgbd/sunrgbd_000017_infos.pkl")
    call_args["out_dir"] = "data/"
    call_args["show"] = False
    call_args["pred_score_thr"] = 0.7

    img = "data/sunrgbd/000017.jpg"
    infos = "data/sunrgbd/sunrgbd_000017_infos.pkl"

    # inferencer = MonoDet3DInferencer(**init_args)
    # inferencer(**call_args)

    model = init_model(init_args['model'], init_args['weights'])
    result = inference_mono_3d_detector(model, img, infos, cam_type='CAM0')

    bboxes3d = result.pred_instances_3d.bboxes_3d.cpu()

    visualizer = Det3DLocalVisualizer()
    vis_img = mmcv.imread(img)
    visualizer.set_image(vis_img)
    meta = {"depth2img" : result.depth2img}
    visualizer.draw_proj_bboxes_3d(bboxes3d, meta)

    visualizer.show()
