from mmcv.image import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import register_all_modules


def initialize(cfg_path, ckpt_path, device):
    register_all_modules()
    model = init_model(cfg_path, ckpt_path, device)

    return model

def inference(input_path, model):
    batch_results = inference_topdown(model, input_path)
    results = merge_data_samples(batch_results)

    return results

def visualize(input_path, output_path, model, results):
    model.cfg.visualizer.radius = 4
    model.cfg.visualizer.alpha = 0.9
    model.cfg.visualizer.line_width = 1

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta, skeleton_style='mmpose')

    input_img = imread(input_path, channel_order='rgb')

    visualizer.add_datasample(
        'result',
        input_img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=False,
        kpt_thr=0.9,
        draw_heatmap=False,
        show_kpt_idx=True,
        skeleton_style='mmpose',
        show=False,
        out_file=output_path)
