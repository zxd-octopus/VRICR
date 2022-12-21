python 	-u main.py \
  		--gpu \
   		--train_batch_size 48 \
    	--dataset Redial  \
      	--task   generation \
       	--ckpt data/ckpt/ckpt_redial_rec.model.ckpt \
       	--data_processed