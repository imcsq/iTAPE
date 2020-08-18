onmt_train -save_model models/iTAPE \
           -data data/data \
           -copy_attn \
           -reuse_copy_attn \
           -copy_loss_by_seqlength \
           -global_attention mlp \
           -word_vec_size 100 \
           -rnn_size 512 \
           -layers 1 \
           -encoder_type brnn \
           -train_steps 25000 \
           -max_grad_norm 2 \
           -dropout 0.0 \
           -batch_size 32 \
           -valid_batch_size 32 \
           -valid_steps 5000 \
           -report_every 100 \
           -optim adam \
           -learning_rate 2 \
           -warmup_steps 8000 \
           -decay_method noam \
           -bridge \
           -seed 789 \
           -world_size 1 \
           -gpu_ranks 0 \
           -pre_word_vecs_enc "data/embeddings.enc.pt" \
           -pre_word_vecs_dec "data/embeddings.dec.pt" \
           -save_checkpoint_steps 5000 \
           -share_embeddings

# -copy_attn, -reuse_copy_attn, and -copy_loss_by_seqlength are used to activate copy mechanism.
# -pre_word_vecs_enc and -pre_word_vecs_dec are used to assign initial embedding weights (e.g. obtained from pretrained GloVe).
# -optim, -learning_rate, -warmup_steps, and decay_method are used to specify the effective and stable optimizer.
