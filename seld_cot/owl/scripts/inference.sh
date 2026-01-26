#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

SLAM_DIR=/path/to/OWL
cd $SLAM_DIR
code_dir=seld_cot/owl

audio_encoder_path=/path/to/sage/encoder/
llm_path=/path/to/llm/
llm_dim=llm-dim # specify llm-dim accordingly

stage=stage-name
qa_data_root=/path/to/questions/
reverb_data_root=/path/to/reverb/
anechoic_data_root=/path/to/audioset/

split=test
output_dir=./outputs/llm-name/${stage}. # specify llm-name accordingly
ckpt_path=$output_dir/ckptt_path # specify checkpoint path accordingly
decode_log=$ckpt_path/decode_${split}_beam4


# -m debugpy --listen 5678 --wait-for-client
python -u $code_dir/inference_seld_batch.py \
        --config-path "conf" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=Llama-2-7b \
        ++model_config.llm_path=$llm_path \
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
        ++dataset_config.fix_length_audio=64 \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=OWL \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=1 \
        ++train_config.output_dir=$output_dir \
        ++train_config.use_peft=true \
        ++peft_config.peft_method=lora \
        ++log_config.log_file=$output_dir/test.log \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt