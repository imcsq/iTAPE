# build aligned dataset and vocab for training
onmt_preprocess -train_src data/body.train.txt \
                -train_tgt data/title.train.txt \
                -valid_src data/body.valid.txt \
                -valid_tgt data/title.valid.txt \
                -save_data data/data \
                -src_seq_length 10000 \
                -tgt_seq_length 10000 \
                -src_seq_length_trunc 400 \
                -tgt_seq_length_trunc 100 \
                -dynamic_dict \
                -share_vocab \
                -shard_size 100000 \
                -num_threads 1

# following step requires the source codes of OpenNMT. this step will generate initial embedding weights based on glove vectors.
OpenNMT-py/tools/embeddings_to_torch.py -emb_file_both "glove_dir/glove.6B.100d.txt" -dict_file "data/data.vocab.pt" -output_file "data/embeddings"
