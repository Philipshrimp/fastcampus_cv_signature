import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.datasets.demo_loader import DemoLoader
from vggsfm.utils.utils import seed_all_random_engines


@hydra.main(config_path="cfgs/", config_name="demo")
def demo(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    print("Model config: ", OmegaConf.to_yaml(cfg))

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    seed_all_random_engines(cfg.seed)

    vggsfm_runner = VGGSfMRunner(cfg)

    test_dataset = DemoLoader(
        SCENE_DIR=cfg.SCENE_DIR,
        img_size=cfg.img_size,
        normalize_cameras=False,
        load_gt=cfg.load_gt,
    )

    sequence_list = test_dataset.sequence_list
    seq_name = sequence_list[0]

    batch, image_paths = test_dataset.get_data(sequence_name=seq_name, return_path=True)
    output_dir = batch["scene_dir"]

    images = batch["image"]
    masks = batch["masks"] if batch["masks"] is not None else None
    crop_params = (batch["crop_params"] if batch["crop_params"] is not None else None)

    original_images = batch["original_images"]

    predictions = vggsfm_runner.run(
        images,
        masks=masks,
        original_images=original_images,
        image_paths=image_paths,
        crop_params=crop_params,
        seq_name=seq_name,
        output_dir=output_dir
    )

if __name__=="__main__":
    with torch.no_grad():
        demo()
