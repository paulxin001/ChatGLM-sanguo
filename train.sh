PRE_SEQ_LEN=8
LR=1e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file data_seed.csv \
    --validation_file data_seed.csv \
    --prompt_column question \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR-dev \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 3333 \
    --logging_steps 10 \
    --save_steps 1111 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

