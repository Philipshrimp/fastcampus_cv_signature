import cv2
import numpy as np
import os
from PIL import Image
from queue import Queue
import threading
import time
import traceback

import mujoco
import mujoco.viewer
import torch
import torch.cuda
from transformers import AutoProcessor, AutoModelForVision2Seq


def setup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def load_mujoco_model(xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Model file not found: {xml_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        return model, data
    except Exception as e:
        raise RuntimeError(f"Error loading MuJoCo model: {e}")
    
def setup_mujoco_renderer(model, image_height, image_width):
    try:
        renderer = mujoco.Renderer(model, height=image_height, width=image_width)
        return renderer
    except Exception as e:
        raise RuntimeError(f"Error setting up MuJoCo renderer: {e}")
    
def setup_mujoco_viewer(model, data):
    try:
        viewer = mujoco.viewer.launch_passive(model, data)

        viewer.cam.azimuth = 90
        viewer.cam.elevation = -10
        viewer.cam.distance = 2.0

        return viewer
    except Exception as e:
        print(f"Could not launch passive viewer: {e}. Proceeding without visualization.")
        return None
    
def load_vla_model(model_name, mujoco_model):
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            max_memory={0: "4Gib"},
            offload_folder="offload")
        model.eval()

        if hasattr(model, "half"):
            model = model.half()

        if not hasattr(model.config, "action_dim"):
            print(f"Warning: Model {model_name} does not have an action dimension. Use MuJoCo's n_actions instead.")
            model.config.action_dim = mujoco_model.nu
        print(f"Model expected action dimension (from config or MuJoCo model): {model.config.action_dim}")

        n_actions_env = mujoco_model.nu
        print(f"MuJoCo environment action dimension: {n_actions_env}")

        if model.config.action_dim != n_actions_env:
            print(f"Warning: Model action dimension ({model.config.action_dim}) does not match MuJoCo environment action dimension ({n_actions_env}).")

        return model, processor
    except Exception as e:
        raise RuntimeError(f"Error loading VLA model: {e}")
    
def process_image(renderer, data, camera_name):
    renderer.update_scene(data, camera=camera_name)

    image_obs_rgb = renderer.render()
    image_obs_rgb = cv2.cvtColor(image_obs_rgb, cv2.COLOR_BGR2RGB)
    image_obs_rgb = Image.fromarray(image_obs_rgb)

    return image_obs_rgb

def prepare_model_inputs(processor, image, text, device):
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77)
    
    inputs = {k : v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in inputs.items()}

    if "pixel_values" in inputs:
        pixel_values = inputs["pixel_values"]

        if len(pixel_values.shape) == 5:
            pixel_values = pixel_values.squeeze(1)
        
        inputs["pixel_values"] = pixel_values

    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs

def run_vla_inference(model, inputs, action_token_length, result_queue):
    try:
        if "input_ids" not in inputs or "attention_mask" not in inputs or "pixel_values" not in inputs:
            raise ValueError("Missing required input keys in inputs dictionary.")
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    return_dict=True)
                
                action_logits = outputs.logits[:, -action_token_length:, :]
                action_probs = torch.softmax(action_logits, dim=-1)
                action_tokens = torch.argmax(action_probs, dim=-1)

                result_queue.put(("success", action_tokens))
    except Exception as e:
        print(f"Error during VLA inference: {e}")
        traceback.print_exc()
        result_queue.put(("error", str(e)))

def decode_action_tokens(model, tokens):
    action_dim = model.config.action_dim
    num_action_bins = 256
    action_tokens = tokens[0, -action_dim:]

    action_values = (action_tokens.float() / (num_action_bins - 1.0)) * 2.0 - 1.0
    action_np = action_values.detach().cpu().numpy()

    if action_np.shape[0] != action_dim:
        if action_np.shape[0] > action_dim:
            action_np = action_np[:action_dim]
        else:
            padded_action = np.zeros(action_dim)
            padded_action[:action_np.shape[0]] = action_np
            action_np = padded_action

    return np.clip(action_np, -1.0, 1.0)

def compute_default_action(data, target_pos):
    current_pos = data.site_xpos[0]
    direction = target_pos - current_pos
    direction = direction / (np.linalg.norm(direction) + 1e-6)

    predicted_action = np.zeros(2)

    predicted_action[0] = direction[0] * 0.5
    predicted_action[1] = direction[1] * 0.5

    return predicted_action

if __name__ == "__main__":
    MUJOCO_MODEL_PATH = "assets/scene.xml"
    OPENVLA_MODEL_NAME = "openvla/openvla-7b"
    RENDER_CAMERA_NAME = "fixed_camera"

    LANGUAGE_INSTRUCTION = "Reach the red target"

    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256

    SIM_DURATION_SECONDS = 20
    CONTROL_FREQ = 10
    RENDER_FREQ = 30

    print("Initializing...")
    device = setup_cuda()
    print(f"Using device: {device}")

    model, data = load_mujoco_model(MUJOCO_MODEL_PATH)
    vla_model, processor = load_vla_model(OPENVLA_MODEL_NAME, model)
    renderer = setup_mujoco_renderer(model, IMAGE_HEIGHT, IMAGE_WIDTH)
    viewer = setup_mujoco_viewer(model, data)

    control_timestep = 1.0 / CONTROL_FREQ
    render_timestep = 1.0 / RENDER_FREQ

    print(f"Simulation timestep: {model.opt.timestep: .4f} seconds")
    print(f"Contorl timestep: {control_timestep: .4f} seconds")
    print(f"Render timestep: {render_timestep: .4f} seconds")

    sim_time = 0.0
    control_step_counter = 0
    render_step_counter = 0
    last_control_time = time.time()
    last_render_time = time.time()

    print("Starting simulation...")

    try:
        mujoco.mj_resetData(model, data)
        start_time = time.time()

        while sim_time < SIM_DURATION_SECONDS:
            current_loop_time = time.time()

            if sim_time == 0 or current_loop_time - last_control_time > control_timestep:
                last_control_time = current_loop_time

                image = process_image(renderer, data, RENDER_CAMERA_NAME)
                inputs = prepare_model_inputs(processor, image, LANGUAGE_INSTRUCTION, device)

                action_token_length = vla_model.config.action_dim * 2
                result_queue = Queue()

                inference_thread = threading.Thread(
                    target=run_vla_inference,
                    args=(vla_model, inputs, action_token_length, result_queue))
                
                inference_thread.daemon = True
                inference_thread.start()

                timeout = 120.0
                inference_thread.join(timeout=timeout)

                if inference_thread.is_alive():
                    print("Inference thread timed out. Terminating...")
                    predicted_action = compute_default_action(data, np.array([0.3, 0.1, 0.0]))
                else:
                    status, result = result_queue.get()

                    if status == "error":
                        print(f"Error during VLA inference: {result}")
                        predicted_action = compute_default_action(data, np.array([0.3, 0.1, 0.0]))
                    else:
                        predicted_action = decode_action_tokens(vla_model, result)

                predicted_action = np.clip(predicted_action, -1.0, 1.0)
                data.ctrl[:] = predicted_action
                control_step_counter += 1

            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep

            if viewer and viewer.is_running():
                if current_loop_time - last_render_time > render_timestep:
                    viewer.sync()

                    last_render_time = current_loop_time
                    render_step_counter += 1
            elif viewer and not viewer.is_running():
                break

    except KeyboardInterrupt:
        print("Simulation interrupted by user. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if viewer and viewer.is_running():
            viewer.close()

        end_time = time.time()

        print("Simulation complete.")
        print(f"Total simulation time: {sim_time: .2f} seconds")
        print(f"Total real time: {end_time - start_time: .2f} seconds")
        print(f"Control steps: {control_step_counter}")
        print(f"Render steps: {render_step_counter}")
                        