# InstructMorpheus-Robot

This work presents an instruction-conditioned multimodal fine-tuning framework that adapts pretrained diffusion models to robotic visual prediction tasks by dynamically aligning textual instructions with visual regions in input frames via cross-attention mechanisms.

## Part.0 environment settings

- computing platform：AutoDL
- mirror：pytorch2.10, cuda12.1

follow [RoboTwin installing instruction]([RoboTwin/INSTALLATION.md at main · TianxingChen/RoboTwin](https://github.com/TianxingChen/RoboTwin/blob/main/INSTALLATION.md)) to download and install RoboTwin in corresponding folder which we have created.

unzip project.zip file, run following command to install RoboTwin virtual environment.

 ```bash
conda activate RoboTwin
pip install -r requirements.txt
 ```

## Part.1 data generation

Make sure your generated data are put  in the `./auto-tmp`folder.

Run following command to activate virtual render.

```bash
sudo apt-get update && sudo apt-get install -y xvfb x11-utils
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99
```

modify RoboTwin config settings in 

```yaml
task_name: block_hammer_beat
render_freq: 0
eval_video_log: false
use_seed: false
collect_data: true
save_path: /root/autodl_tmp
dual_arm: true
st_episode: 0
head_camera_type: D435
wrist_camera_type: D435
front_camera_type: D435
pcd_crop: true
pcd_down_sample_num: 1024
episode_num: 100
save_freq: 15
save_type:
  raw_data: false
  pkl: true
data_type:
  rgb: true
  observer: false
  depth: false
  pointcloud: false
  endpose: true
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
```

block_handover

```yaml
task_name: block_handover
render_freq: 0
eval_video_log: false
use_seed: false
collect_data: true
save_path: /root/autodl_tmp
dual_arm: true
st_episode: 0
head_camera_type: D435
wrist_camera_type: D435
front_camera_type: D435
pcd_crop: true
pcd_down_sample_num: 1024
episode_num: 100
save_freq: 15
save_type:
  raw_data: false
  pkl: true
data_type:
  rgb: true
  observer: false
  depth: false
  pointcloud: false
  endpose: true
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
```

blocks_stack_easy

```yaml
task_name: blocks_stack_easy
render_freq: 0
eval_video_log: false
use_seed: false
collect_data: true
save_path: /root/autodl_tmp
dual_arm: true
st_episode: 0
head_camera_type: D435
wrist_camera_type: D435
front_camera_type: D435
pcd_crop: true
pcd_down_sample_num: 1024
episode_num: 100
save_freq: 15
save_type:
  raw_data: false
  pkl: true
data_type:
  rgb: true
  observer: false
  depth: false
  pointcloud: false
  endpose: true
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
```

make sure you are in the project folder.

following  [RoboTwin installing instruction]([RoboTwin/INSTALLATION.md at main · TianxingChen/RoboTwin](https://github.com/TianxingChen/RoboTwin/blob/main/INSTALLATION.md)) to generate following .pkl data:
```bash
"/autodl-tmp/block_hammer_beat_D435_pkl"
"/autodl-tmp/block_handover_D435_pkl"
"/autodl-tmp/blocks_stack_easy_D435_pkl"
```

or you can just run our bash script generate all the data we need.

```bash
cd project/path
./run_all_task.sh
```

## Part.2 install instruct-pix2pix

(we have already provide the instruct-pix2pix folder, so you can go Part.3 directly if you don't want to download yourself)

Run following command:

```bash
# make sure you are in the path/to/project/folder/autodl-tmp
git clone https://github.com/timothybrooks/instruct-pix2pix.git # clone the original repository
conda env create -f environment.yaml # set up conda environment
cd instruct-pix2pix
```

modify `config/train.yaml`:

```yaml
model:
  base_learning_rate: 1.0e-04
  target: ldm.models.diffusion.ddpm_edit.LatentDiffusion
  params:
    ckpt_path: stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: edited
    cond_stage_key: edit
    image_size: 32
    channels: 4
    cond_stage_trainable: false   # Note: different from the one we trained before
    conditioning_key: hybrid
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: true
    load_ema: false

    scheduler_config: # 10000 warmup steps
      target: ldm.lr_scheduler.LambdaLinearScheduler
      params:
        warm_up_steps: [ 0 ]
        cycle_lengths: [ 10000000000000 ] # incredibly large number to prevent corner cases
        f_start: [ 1.e-6 ]
        f_max: [ 1. ]
        f_min: [ 1. ]

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32 # unused
        in_channels: 8
        out_channels: 4
        model_channels: 320
        attention_resolutions: [ 4, 2, 1 ]
        num_res_blocks: 2
        channel_mult: [ 1, 2, 4, 4 ]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder

data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 32
    num_workers: 2
    train:
      target: edit_dataset.EditDataset
      params:
        path: data/clip-filtered-dataset
        split: train
        min_resize_res: 256
        max_resize_res: 256
        crop_res: 256
        flip_prob: 0.5
    validation:
      target: edit_dataset.EditDataset
      params:
        path: data/clip-filtered-dataset
        split: val
        min_resize_res: 256
        max_resize_res: 256
        crop_res: 256

lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 2000
        max_images: 2
        increase_log_steps: False

  trainer:
    max_epochs: 100 # 2000 to 100
    benchmark: True
    accumulate_grad_batches: 8 # 4 to 8
    check_val_every_n_epoch: 4
```

## Part.3 data preparation for instruct-pix2pix

run following command to generate data pairs and corresponding dataset structure.

```bash
cd project/path
python dataset_gen.py
```

dataset path tree:

```bash
autodl-tmp/instruct-pix2pix/data/instruct-pix2pix-dataset-000/
├── 0000000/
│   ├── prompt.json
│   ├── 000000_0.jpg     
│   └── 000000_1.jpg
├── 0000001/
│   ├── prompt.json
│   └── 000001_0.jpg
│   └── 000001_1.jpg
└── ...
```

generate seeds.json for training:

```bash
cd autodl-tmp/instruct-pix2pix
python dataset_creation/prepare_dataset.py data/instruct-pix2pix-dataset-000
```

## Part.4 Start training the model

our code only runs on dual GPU settings, before continue, make sure your device have dual GPU available.

run following command to start training:

```bash
python main.py --name default --base configs/train.yaml --train --gpus 0,1
```

![image](https://github.com/user-attachments/assets/ef17de53-4cc7-4634-9daa-2f423c09d035)
![image](https://github.com/user-attachments/assets/3c48dbbe-aa88-49ae-9d00-ded78f6bb8a2)


## Part.5 evalutaion

run following command to generate evalutaion results and predictions.

```bash
python eval.py --ckpt logs/train_default/checkpoints/last.ckpt
```

## Part.6 Experimental Results

![image](https://github.com/user-attachments/assets/4365bda0-41f0-4262-aaeb-c467e03f5825)
![image](https://github.com/user-attachments/assets/241dfd2a-5eeb-4dd7-bf94-e32e137a619c)
![image](https://github.com/user-attachments/assets/2e51afae-2862-4250-a5b5-8e44b481b68d)

