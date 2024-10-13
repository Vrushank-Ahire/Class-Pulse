from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np

model = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess(image):
    # Implement preprocessing steps (resize, normalize, etc.)
    return image

def get_face_embedding(image):
    img_cropped = preprocess(image)  # Preprocess image (crop, resize)
    embedding = model(img_cropped.unsqueeze(0))  # Get embedding
    return embedding

def compare_embeddings(embedding1, embedding2, threshold=0.5):
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance < threshold