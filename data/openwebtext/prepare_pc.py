# data/openwebtext/prepare_pc.py
"""
Saves the OpenWebText dataset to binary files for training, with download progress.
"""

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import multiprocessing
from huggingface_hub import HfFileSystem, hf_hub_download

# -----------------------------------------------------------------------------
# Configuration
num_proc = multiprocessing.cpu_count() // 2
num_proc_load_dataset = num_proc

# -----------------------------------------------------------------------------

def download_with_progress(dataset_name):
    """
    Custom function to download dataset files with progress bars.
    Returns the loaded dataset.
    """
    print(f"Preparing to download {dataset_name} dataset...")
    
    # Use HfFileSystem to list files in the dataset repository
    fs = HfFileSystem()
    dataset_path = f"datasets/{dataset_name}"
    files = fs.ls(dataset_path, detail=False)
    
    # Filter for data files (e.g., .arrow, .parquet, etc.)
    data_files = [f for f in files if f.endswith((".arrow", ".parquet", ".jsonl"))]
    
    # Download each file with progress
    local_files = []
    for file in tqdm(data_files, desc="Downloading dataset files"):
        local_file = hf_hub_download(
            repo_id=dataset_name,
            filename=file.split(f"{dataset_name}/")[-1],
            repo_type="dataset",
            cache_dir=None,  # Use default cache
        )
        local_files.append(local_file)
    
    # Load the dataset from downloaded files
    print("Loading dataset from downloaded files...")
    dataset = load_dataset(
        dataset_name,
        num_proc=num_proc_load_dataset,
        trust_remote_code=True,
        data_files={"train": local_files},  # Explicitly use downloaded files
        download_mode="reuse_dataset_if_exists",
    )
    return dataset

if __name__ == "__main__":
    try:
        print("Starting script...")
        print("Current working directory:", os.getcwd())

        # Load the OpenWebText dataset with progress
        print("Loading OpenWebText dataset with progress...")
        dataset = download_with_progress("openwebtext")
        print("Dataset loaded successfully. Keys:", dataset.keys())

        # Create train/val split
        print("Splitting dataset into train and validation sets...")
        split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')
        print("Dataset split completed. Train size:", len(split_dataset['train']), "Val size:", len(split_dataset['val']))

        # Tokenize function with its own tokenizer instance
        def process(example):
            enc = tiktoken.get_encoding("gpt2")  # Initialize tokenizer inside the function
            ids = enc.encode_ordinary(example['text'])
            ids.append(enc.eot_token)
            return {'ids': ids, 'len': len(ids)}

        # Tokenize both splits
        for split in ['train', 'val']:
            print(f"Tokenizing {split} split...")
            tokenized = split_dataset[split].map(
                process,
                remove_columns=['text'],
                desc=f"Tokenizing {split}",
                num_proc=num_proc,
            )
            print(f"{split} split tokenized. Total tokens:", sum(tokenized['len']))

            # Concatenate tokens
            print(f"Concatenating tokens for {split} split...")
            total_length = sum(tokenized['len'])
            ids = np.zeros(total_length, dtype=np.uint16)
            offset = 0
            for example in tqdm(tokenized, desc=f"Building {split} array"):
                ids[offset:offset + example['len']] = example['ids']
                offset += example['len']
            print(f"{split} tokens concatenated. Length:", len(ids))

            # Save to file
            output_file = f"{split}.bin"
            print(f"Saving {split} data to {output_file}...")
            ids.tofile(output_file)
            print(f"{split} data saved. File size (bytes):", os.path.getsize(output_file))

        print("Done! Training and validation data saved as train.bin and val.bin")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise