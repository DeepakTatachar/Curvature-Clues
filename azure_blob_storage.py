"""
Author: Deepak Ravikumar Tatachar
Copyright © 2024 Deepak Ravikumar, Nano(Neuro) Electronics Research Lab, Purdue University
"""

import os
from azure.storage.blob import BlobServiceClient
import io
import torch
import numpy as np
import json
import pickle

def download_blob_to_stream(blob_service_client: BlobServiceClient, container_name: str, blob: str):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob)

    # readinto() downloads the blob contents to a stream and returns the number of bytes read
    stream = io.BytesIO()
    num_bytes = blob_client.download_blob().readinto(stream)
    stream.seek(0)
    return stream

def get_connection_string():
    with open("config.json", 'r') as f:
        data = json.load(f)
    
    return data['connection_string']

def upload_blob_file(blob_service_client: BlobServiceClient, container_name: str, path, blob_name, file_name):
    container_client = blob_service_client.get_container_client(container=container_name)
    with open(file=os.path.join(path, file_name), mode="rb") as data:
        blob_client = container_client.upload_blob(name=blob_name, data=data, overwrite=True)

def get_model_from_azure_blob_file(container_name, container_file_name):
    connect_str = get_connection_string()

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    print(f'Getting model {container_file_name} from {container_name}')
    buffer = download_blob_to_stream(blob_service_client, container_name, container_file_name)
    model_params = torch.load(buffer, map_location='cpu')
    return model_params

def cloud_save(model_state_dict, path, gpu_id):
    scratch_name = f"{os.uname()[1]}_{gpu_id}"
    torch.save(model_state_dict, f"./data/temp{scratch_name}.up")
    name = os.path.split(path)[-1]
    dataset = name.split("_")[0]

    serialized_bytes = torch_state_dict_to_bytes(model_state_dict)
    connect_str = get_connection_string()

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Create a unique name for the container
    container_name = "curvature-mi-models"

    if name.split(".")[-1] == "temp":
        container_file_name = f"{dataset}/temp/{name}"
    else:
        # Create a file in the local data directory to upload and download
        container_file_name = f"{dataset}/{name}"
    
    upload_blob_file(blob_service_client, container_name, "./data", container_file_name, f"temp{scratch_name}.up")

def get_model_from_azure_blob(dataset='cifar100', seed=0):
    if dataset == 'imagenet':
        return get_model_from_azure_blob_fz_imagenet_resnet(seed=seed)

    connect_str = get_connection_string()

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Create a unique name for the container
    container_name = "curvature-mi-models"

    # Create a file in the local data directory to upload and download
    container_file_name = f"{dataset}/{dataset}_resnet18_ds{seed}_shadow.ckpt"

    buffer = download_blob_to_stream(blob_service_client, container_name, container_file_name)
    model_params = torch.load(buffer, map_location='cpu')

    return model_params

def torch_state_dict_to_bytes(state_dict):
    """Converts a PyTorch state dictionary to bytes for upload to Azure Blob Storage.

    Args:
        state_dict: The PyTorch state dictionary to convert.

    Returns:
        bytes: The serialized state dictionary as bytes.

    Raises:
        ValueError: If the state dictionary contains unsupported data types.
    """

    # Create a buffer to hold the serialized state dictionary
    buffer = io.BytesIO()

    # Serialize the state dictionary using pickle, addressing encoding consistency
    pickle.dump(state_dict, buffer, protocol=pickle.HIGHEST_PROTOCOL)

    # Seek to the beginning of the buffer and return the serialized bytes
    buffer.seek(0)
    return buffer.read()

def get_model_from_azure_blob_fz_imagenet_resnet(seed=0):
    dataset = 'imagenet'
    connect_str = get_connection_string()

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Create a unique name for the container
    container_name = "curvature-mi-models"

    # Create a file in the local data directory to upload and download
    container_file_name = f"{dataset}/{dataset}_resnet50_{seed}.ckpt"

    buffer = download_blob_to_stream(blob_service_client, container_name, container_file_name)
    model_params = torch.load(buffer, map_location='cpu')

    return model_params

def upload_numpy_as_blob(container_name, container_dir, file_name, numpy_array, overwrite=False):

    # Create a BytesIO stream object in memory
    stream = io.BytesIO()

    # Use a memory buffer and array serialization directly to the stream
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=numpy_array)
    stream.write(buffer.getvalue())

    # Seek to the beginning of the stream for reading
    stream.seek(0)
    connect_str = get_connection_string()

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=f"{container_dir}/{file_name}")
    blob_client.upload_blob(stream.read(), blob_type="BlockBlob", overwrite=overwrite)

def get_numpy_from_azure(container_name, container_dir, file_name):

    connect_str = get_connection_string()

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Create a file in the local data directory to upload and download
    container_file_name = f"{container_dir}/{file_name}"

    buffer = download_blob_to_stream(blob_service_client, container_name, container_file_name)
    buffer.seek(0)  # Ensure we start from the beginning
    loaded_array = np.load(buffer)

    if 'data' in loaded_array:
        return loaded_array['data']

    return loaded_array