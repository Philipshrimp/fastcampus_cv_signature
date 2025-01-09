import cv2
import json
import numpy as np
import os
from PIL import Image
from tqdm.notebook import tqdm

import faiss
import torch
import torchvision.transforms as T


transform_image = T.Compose([
    T.ToTensor(),
    T.Resize(224),
    T.CenterCrop(224),
    T.Normalize([0.5], [0.5])])

def load_image(img_path) -> torch.Tensor:
    input_img = Image.open(img_path)
    transformed_img = transform_image(input_img)[:3].unsqueeze(0)

    return transformed_img

def create_index(files, model) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(384)
    all_embeddings = {}

    with torch.no_grad():
        for _, file in enumerate(tqdm(files)):
            embeddings = model(load_image(file).to(device))
            embedding = embeddings[0].cpu().numpy()

            reshaped_embedding = np.array(embedding).reshape(1, -1)

            all_embeddings[file] = reshaped_embedding.tolist()
            index.add(reshaped_embedding)

    with open("all_embeddings.json", "w") as file:
        file.write(json.dumps(all_embeddings))

    faiss.write_index(index, "data.bin")

    return index, all_embeddings

def search_index(input_index, input_embeddings, k=3) -> list:
    _, results = input_index.search(np.array(input_embeddings[0].reshape(1, -1)), k)

    return results[0]

if __name__=="__main__":
    cwd = os.getcwd()

    ROOT_DIR = os.path.join(cwd, "COCO-128-2/train/")

    files = os.listdir(ROOT_DIR)
    files = [os.path.join(ROOT_DIR, f) for f in files if f.lower().endswith(".jpg")]

    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_vits14.to(device)

    data_index, all_embeddings = create_index(files, dinov2_vits14)

    input_file = "COCO-128-2/valid/000000000081_jpg.rf.5262c2db56ea4568d7d32def1bde3d06.jpg"
    input_img = cv2.resize(cv2.imread(input_file), (416, 416))

    with torch.no_grad():
        embedding = dinov2_vits14(load_image(input_file).to(device))
        results = search_index(data_index, np.array(embedding[0].cpu()).reshape(1, -1))

        for i, index in enumerate(results):
            print(f"Image {i}: {files[index]}")
