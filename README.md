# Code for iTAPE

This is the source code for the paper **"Stay professional and efficient: automatically generate titles for your bug reports"**.

## Package Requirement

To run this code, some packages are needed as following:

```
opennmt-py >= 1.0.0
pytorch >= 1.3.0
nltk >= 3.4.5
```

## Prepare dataset for OpenNMT

We provided our preprocessed TXT dataset in `data` folder for the replication, which is obtained by applying all three sample refinement rules and inserting tag tokens around each human-named tokens. If you would like to create **a new dataset** by applying different sample refining rules or another human-named token hinting method on all the collected issue samples, please follow the instructions in this section.

### Preprocess from raw data

We collect 922730 issue samples from GitHub and store them in a JSON file. Each sample contains 4 fields, i.e. `repo`, `number`, `body`, and `title`. *This data could be found at [here](https://drive.google.com/file/d/1UXzMf61KLYCifjbk5vWnzKi8aYr0VMiy/view?usp=sharing).*

`0-0-preprocess_and_refine.py` is provided to filter out the samples and data which is not suitable for later training.

This step will first carry preprocessing on the sample set to remove samples hard to tokenize and tailor miscellaneous content in data. Afterward, it applies three heuristic rules to remove samples with unsuitable titles to build a fairly reliable dataset.

### Generate TXT data file for OpenNMT

After obtaining suitable samples, `0-1-export_txtdata.py` is provided to transform JSON data file into TXT file for later processing with the general summarization model in OpenNMT.

This step will first extract and process the two types of human-named tokens, i.e. identifiers and version numbers, then transform the text into lowercased token sequences and export data in TXT files.

## Perform general summarization with OpenNMT

This section introduces how to use the general summarization model in OpenNMT to perform the summarization task with the prepared data in TXT files.

### Train

To train the model, the first step is to build an src-tgt aligned dataset and a vocab. `1-build.sh` is provided to do these necessary preparations for data in TXT format.

To obtain GloVe as initial word embeddings, please refer to the tutorial in [OpenNMT FAQ](https://opennmt.net/OpenNMT-py/FAQ.html).

Then, we can perform training with OpenNMT. The training command and parameter configuration are provided in `2-train.sh`. Run this script will train an one-layer LSTM-based Seq2Seq summarization model with Copy Mechanism that is activated by using `-copy_attn` option.

### Test

To generate titles for issue bodies in the testing set, `3-test.sh` is provided to apply the trained model for this step. After running this script, the generated titles will be orderly saved into a TXT file in `testout` directory.

## References

For more details about data processing, please refer to the `code comments` and our paper "Stay professional and efficient: automatically generate titles for your bug reports".

For more flexible and specific parameter settings during performing general summarization, please refer to the tutorial of [OpenNMT](https://opennmt.net/OpenNMT-py/). This project highly appreciates OpenNMT to provide a flexible and easy-used general summarization tool.
