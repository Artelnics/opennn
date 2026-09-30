# BERT example data and pretrained model

## Amazon reviews data

Kotzias, D. (2015), *Sentiment Labelled Sentences*.
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences),
DOI [10.24432/C57604](https://doi.org/10.24432/C57604).
Kotzias, D., Denil, M., de Freitas, N., Smyth, P. (2015), *From Group to
Individual Labels using Deep Features*, KDD 2015.

Data licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Preserve the attribution, source and licence link when redistributing.

OpenNN adaptation: `amazon_cells_labelled.txt` holds the 1,000 labelled Amazon
sentences with the 0/1 labels mapped to Bad/Good. A leading apostrophe was
added to the first sentence.

## BERT-Base Uncased pretrained model

The example downloads Google's BERT-Base Uncased encoder in OpenNN format
from [Artelnics/bert-base-uncased-opennn](https://huggingface.co/Artelnics/bert-base-uncased-opennn).
The identical files were originally published as `bert-weights-v1` on
2026-07-07 and migrated to Hugging Face on 2026-09-30.
The weights and vocabulary are downloaded at run time;
they are not committed to OpenNN or included in its library installation package.

### Original source and licence

Google's [original model release](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/README.md#pre-trained-models)
explicitly licenses the pretrained models under **Apache License 2.0**, the
same licence as its source code. The original BERT-Base Uncased checkpoint is
`uncased_L-12_H-768_A-12`, released on 2018-10-18. Copyright attribution in the
upstream implementation: **Copyright 2018 The Google AI Language Team Authors.**

`LICENSE_BERT.txt` is a byte-for-byte copy of the
[upstream licence](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/LICENSE).
`NOTICE_BERT.txt` records the original attribution and OpenNN's adaptations;
it is an OpenNN notice, not a file supplied by the original release. The
upstream repository and the Hugging Face model revision inspected below have
no separate `NOTICE` file.

Apache 2.0 permits use, modification and redistribution, including commercial
use. Redistribution requires a copy of the licence, preservation of relevant
copyright, patent, trademark and attribution notices, and prominent notices
of changes in modified files. Its patent and trademark provisions also apply.
Include `LICENSE_BERT.txt`, `NOTICE_BERT.txt` and this source record with
redistributed weights or vocabulary. OpenNN's LGPL software licence does not
replace the model's Apache 2.0 terms or the reviews' CC BY 4.0 terms.

Credit: Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova,
[*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://aclanthology.org/N19-1423/),
NAACL-HLT 2019, pages 4171-4186, DOI 10.18653/v1/N19-1423.

### Conversion and file identification

The converted release targets `BertForSequenceClassification` with dimensions
`64, 30522, 768, 12, 3072, 12, 1`: a sequence length of 64, the original
30,522-token vocabulary, 12 encoder layers and a single-output classification
head. The binary storage and classification architecture are OpenNN
adaptations. The example fine-tunes the model on the Amazon reviews above.

The released vocabulary matches byte-for-byte the original model's
[`vocab.txt` at google-bert/bert-base-uncased revision 86b5e0934494bd15c9632b12f734a8a67f723594](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594).
All 30,522 x 768 word-embedding values at the start of the converted binary
also match that revision's `model.safetensors` exactly. This comparison
identifies the base model; it does not assert that every converted parameter
or the classification head equals an upstream tensor. The conversion release
does not record the exact upstream revision used by its converter.

Both release assets were downloaded and checked against their GitHub-provided
SHA-256 digests on 2026-09-30:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `bert-base-uncased-seq64.bin` | 436558912 | `bc9e3252f393ccf2211f9f98240352d5af69a1654c5f651e461c360a9efc6e73` |
| `bert-base-uncased-vocab.txt` | 231508 | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |

Download base URL:
`https://huggingface.co/Artelnics/bert-base-uncased-opennn/resolve/main/`.

`LICENSE_BERT.txt` SHA-256:
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.
The upstream `model.safetensors` used for the comparison has SHA-256
`68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`.

### Running and redistributing the example

CMake copies the bundled reviews and all three notices to `../data/bert`.
Run `bert [reviews.txt] [model_dir]` from the executable directory; it trains
for three epochs with batches of 32 and then prints binary classification
metrics. The default model cache is `../data/bert` and the first run downloads
about 437 MB. If a different cache is used or the model files are downloaded
separately, keep the model notices beside those files.

The Hugging Face repository includes the complete licence, model notice and
source record, with links from its model card. Preserve these notices with
redistributed copies of the model and vocabulary.
