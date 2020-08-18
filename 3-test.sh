nohup onmt_translate -gpu 0  \
               -batch_size 32 \
               -beam_size 10 \
               -model models/iTAPE_step_25000.pt \
               -src data/body.test.txt \
               -output testout/iTAPE_step_25000_minlen8.txt \
               -min_length 8 \
               -stepwise_penalty \
               -length_penalty wu \
               -alpha 0.9 \
               -replace_unk \
               -block_ngram_repeat 2

# -min_length is used to set the minimum output threshold (toks).
# -beam_size is used to activate beam search.
# to avoid generation invalid <unk> in our output target, which has high requirement on accurateness, we activate -replace_unk.
# to avoid useless repeat in our output target, which has high requirement on conciseness, we acitave -block_ngram_repeat.
