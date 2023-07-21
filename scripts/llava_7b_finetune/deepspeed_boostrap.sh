MODEL_HUB_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-nlpml/wangshichao10/model_hub
MLLM_DATA_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-nlpml/wangshichao10/mllm/data

SCRIPT_DIR=$(dirname "$0")

# deepspeed \
torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    llava/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/zero2.json \
    --model_name_or_path $MODEL_HUB_HOME/huggingface/lmsys/vicuna-7b-v1.3 \
    --version v1 \
    --data_path $MLLM_DATA_HOME/LLaVA-Instruct-150K/llava_instruct_150k.json \
    --image_folder $MLLM_DATA_HOME/train2017 \
    --vision_tower $MODEL_HUB_HOME/huggingface/openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/llava-7b-v1-pretrained.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --output_dir ./checkpoints/llava-7b-v1-finetuned \
    --fp16 True \
    --bf16 False \
    --tf32 False \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb