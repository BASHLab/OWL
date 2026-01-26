#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

SLAM_DIR=/path/to/OWL
cd $SLAM_DIR
code_dir=seld_cot/owl

audio_encoder_path=/path/to/sage/encoder/
llm_path=/path/to/llm/

stage=stage-name
qa_data_root=/path/to/questions/
reverb_data_root=/path/to/reverb/
anechoic_data_root=/path/to/audioset/
ckpt_path=/path/to/checkpoints/  # optional, for loading existing checkpoints

split=train
llm=llm-name
llm_dim=llm-dim  

output_dir=./outputs/${llm}/${stage}


hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=$llm \
++model_config.llm_path=$llm_path \
++model_config.ckpt_path=$ckpt_path \
++model_config.llm_dim=$llm_dim \
++model_config.encoder_name=SAGE \
++model_config.encoder_projector=q-former \
++model_config.qformer_layers=8 \
++model_config.encoder_ckpt=$audio_encoder_path \
++dataset_config.test_split=${split} \
++dataset_config.stage=$stage \
++dataset_config.qa_data_root=$qa_data_root \
++dataset_config.anechoic_data_root=$anechoic_data_root \
++dataset_config.reverb_data_root=$reverb_data_root \
++dataset_config.max_words=96 \
++dataset_config.fix_length_audio=64 \
++train_config.model_name=OWL \
++train_config.num_epochs=4 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=5000 \
++train_config.total_steps=50000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=2000 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++train_config.use_peft=true \
++peft_config.peft_method=lora \
++metric=acc \
++log_config.log_file=$output_dir/log.txt \
++log_config.use_wandb=true \
++log_config.wandb_dir=$output_dir/wandb \
++log_config.wandb_project_name=${llm}-${stage} \
++log_config.wandb_exp_name=${llm}-${stage} \
++log_config.wandb_entity_name=anonymizegithub-worcester-polytechnic-institute \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -u -m debugpy --listen 55555 --wait-for-client $code_dir/finetune_seld.py \
        --config-path "conf" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 4 \
        --master_port=39503 \
        $code_dir/finetune_seld.py \
        --config-path "conf" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=false \
        $hydra_args
fi
