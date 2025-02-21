from efm3d.inference.pipeline import run_one

if __name__=="__main__":
    input_data = "./data/seq136_sample/video.vrs"
    config_path = "./efm3d/config/evl_inf_desktop.yaml"
    ckpt_path = "./ckpt/model_lite.pth"

    voxel_res = 0.08

    run_one(
        data_path=input_data,
        model_ckpt=ckpt_path,
        model_cfg=config_path,
        voxel_res=voxel_res,
    )