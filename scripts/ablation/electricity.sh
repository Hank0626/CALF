export CUDA_VISIBLE_DEVICES=5

seq_len=96
model=GPT4TS

for pred_len in 96
do
    for task_w in 1
    do
        for feature_w in 0 0.01
        do
            for logits_w in 0 1
            do
                if [ "$task_w" = "0" ] && [ "$feature_w" = "0" ] && [ "$logits_w" = "0" ];
                then
                    continue
                fi


python run.py \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ECL_$model'_'$seq_len'_'$pred_len \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --train_epochs 20 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 3 \
    --task_loss smooth_l1 \
    --distill_loss smooth_l1 \
    --logits_loss smooth_l1\
    --task_w $task_w \
    --logits_w $logits_w \
    --feature_w $feature_w



echo "+++++++++++++++++++++++++++++++++++++++++++"

            done
        done
    done

echo '====================================================================================================================='
done

