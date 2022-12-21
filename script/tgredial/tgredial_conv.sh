python -u main.py  \
		--gpu  \
		--dataset TG  \
		--task generation  \
		--test_batch_size 128  \
		--train_batch_size 32  \
		--ckpt data/ckpt/ckpt_tgredial_rec.model.ckpt
