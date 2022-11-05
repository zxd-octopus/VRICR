python -u main.py \
          --gpu \
          --dataset TG \
          --task generation \
          --test_batch_size 128 \
          --train_batch_size 32 \
          --eval_interval 800  \
          --ckpt data/ckpt/TG/recommend/{task_ID}/{best_ckpt_path_for_rec_finetune}