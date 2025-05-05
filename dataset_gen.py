import os
import pickle
import json
import cv2
from tqdm import tqdm

# Configuration
datasets = [
    "/root/autodl-tmp/block_hammer_beat_D435_pkl",
    "/root/autodl-tmp/block_handover_D435_pkl",
    "/root/autodl-tmp/blocks_stack_easy_D435_pkl"
]

prompts = ["beat the block with the hammer", "handover the blocks", "stack blocks"]
output_base = "/root/autodl-tmp/instruct-pix2pix/data/instruct-pix2pix-dataset-000"
episode_count = 100  # Assuming each dataset has 100 episodes

def process_pkl_pair(pkl1_path, pkl2_path, output_dir, group_idx, prompt):
    # Load data from both pkl files
    with open(pkl1_path, 'rb') as f:
        data1 = pickle.load(f)
    with open(pkl2_path, 'rb') as f:
        data2 = pickle.load(f)
    
    # Extract RGB arrays
    img1 = data1['observation']['head_camera']['rgb']
    img2 = data2['observation']['head_camera']['rgb']
    
    # Create directory for this group
    group_dir = os.path.join(output_dir, f"{group_idx:07d}")
    os.makedirs(group_dir, exist_ok=True)
    
    # Save images
    img_idx = group_idx  # Using group index as image prefix
    cv2.imwrite(os.path.join(group_dir, f"{img_idx:06d}_0.jpg"), img1[..., ::-1])  # Convert BGR to RGB
    cv2.imwrite(os.path.join(group_dir, f"{img_idx:06d}_1.jpg"), img2[..., ::-1])
    
    # Create prompt.json (customize as needed)
    prompt_data = {
        "edit": prompt
    }
    with open(os.path.join(group_dir, "prompt.json"), 'w') as f:
        json.dump(prompt_data, f)

def process_dataset(dataset_path, start_group_idx, prompt, interval):
    current_group_idx = start_group_idx
    
    for episode_idx in tqdm(range(episode_count), desc=f"Processing {os.path.basename(dataset_path)}"):
        episode_dir = os.path.join(dataset_path, f"episode{episode_idx}")
        
        if not os.path.exists(episode_dir):
            continue
            
        # Get all pkl files in the episode and sort them numerically
        pkl_files = [f for f in os.listdir(episode_dir) if f.endswith('.pkl')]
        pkl_files.sort(key=lambda x: int(x.split('.')[0]))
        
        # Process files in pairs
        # for i in range(0, len(pkl_files)-interval, interval):
        #     pkl1_path = os.path.join(episode_dir, pkl_files[i])
        #     pkl2_path = os.path.join(episode_dir, pkl_files[i+interval])
            
        #     process_pkl_pair(pkl1_path, pkl2_path, output_base, current_group_idx, prompt)
        #     current_group_idx += 1
        pkl1_path = os.path.join(episode_dir, pkl_files[0])
        pkl2_path = os.path.join(episode_dir, pkl_files[-1])
        process_pkl_pair(pkl1_path, pkl2_path, output_base, current_group_idx, prompt)
        current_group_idx += 1
        
    return current_group_idx

def main():
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    global_group_idx = 0
    
    for i, dataset_path in enumerate(datasets):
        print(f"\nStarting dataset: {dataset_path}")
        global_group_idx = process_dataset(dataset_path, global_group_idx, prompts[i], 50)
        print(f"Finished dataset. Next group index: {global_group_idx}")
    
    print(f"\nAll datasets processed! Total groups created: {global_group_idx}")

if __name__ == "__main__":
    main()