
MODEL_HUB_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-nlpml/wangshichao10/model_hub
MLLM_DATA_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-nlpml/wangshichao10/mllm/data

torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path $MODEL_HUB_HOME/huggingface/lmsys/vicuna-7b-v1.3 \
    --version v1 \
    --data_path $MLLM_DATA_HOME/LLaVA-CC3M-Pretrain-595K/chat.json \
    --image_folder $MLLM_DATA_HOME/LLaVA-CC3M-Pretrain-595K/images \
    --vision_tower $MODEL_HUB_HOME/huggingface/openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 False \
    --output_dir $MODEL_HUB_HOME/llava-7b-v1-pretrained_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
