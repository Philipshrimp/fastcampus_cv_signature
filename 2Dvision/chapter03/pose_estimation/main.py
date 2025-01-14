import torch

import model


if __name__=='__main__':
    input_path = 'COCO-128-2/train/000000000625_jpg.rf.ce871c39393fefd9fd8671806761a1c8.jpg'
    output_path = 'result.png'

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    cfg_path = 'configs/yoloxpose_m_8xb32-300e_coco-640.py'
    ckpt_path = 'weights/yoloxpose_m_8xb32-300e_coco-640-84e9a538_20230829.pth'

    estimator = model.initialize(cfg_path, ckpt_path, device)
    results = model.inference(input_path, estimator)

    model.visualize(input_path, output_path, estimator, results)
