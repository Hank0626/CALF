export CUDA_VISIBLE_DEVICES=5

seq_len=96
model=CALF


for pred_len in 96
do
    for task_w in 0 1
    do
        for feature_w in 0 0.01
        do
            for output_w in 0 1
            do
                if [ "$task_w" = "0" ] && [ "$feature_w" = "0" ] && [ "$output_w" = "0" ];
                then
                    continue
                fi

python run.py \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id traffic_$model'_'$seq_len'_'$pred_len \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --task_loss smooth_l1 \
    --feature_loss smooth_l1 \
    --output_loss smooth_l1 \
    --task_w $task_w \
    --output_w $output_w \
    --feature_w $feature_w

echo "+++++++++++++++++++++++++++++++++++++++++++"

            done
        done
    done

echo '====================================================================================================================='
done
