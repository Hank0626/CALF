export CUDA_VISIBLE_DEVICES=4

seq_len=96
model=CALF

for pred_len in 96 192 336 720
do
    for task_w in 1
    do
        for feature_w in 0 0.01
        do
            for output_w in 0 1
            do
                if [ "$task_w" = "1" ] && [ "$feature_w" = "0.01" ] && [ "$output_w" = "1" ];
                then
                    continue
                fi


python run.py \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id weather_$model'_'$seq_len'_'$pred_len \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --lradj type3 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 3 \
    --task_w $task_w \
    --output_w $output_w \
    --feature_w $feature_w

echo "+++++++++++++++++++++++++++++++++++++++++++"

            done
        done
    done

echo '====================================================================================================================='
done

