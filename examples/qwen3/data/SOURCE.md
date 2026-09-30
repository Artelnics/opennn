# Qwen3 pretrained models

The `qwen3` chat example downloads **Qwen3-4B** in OpenNN format. The library
also supports **Qwen3-0.6B** through `Qwen3::Variant::B0_6`; the example's entry
point selects `Qwen3::Variant::B4`. Both original models and both Artelnics
conversions declare **Apache License 2.0**. Qwen weights and its original
pretraining/post-training corpora are not committed to OpenNN or included
in its library installation package.

## Original sources and licence

The original Alibaba Cloud model releases inspected on 2026-09-30 are:

- [Qwen/Qwen3-0.6B at c1899de289a04d12100db370d81485cdf75e47ca](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca).
- [Qwen/Qwen3-4B at 1cfa9a7208912126459214e8b04321603b3df60c](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c).

Both include the same complete Apache 2.0 text, including the original
**Copyright 2024 Alibaba Cloud** notice. `LICENSE_QWEN3.txt` is an unmodified,
byte-for-byte copy of that file, with SHA-256
`832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`.
Keep the original year; do not replace it with the model's release year.

Apache 2.0 permits use, modification and redistribution, including commercial
use. Redistribution requires the licence text, preservation of relevant
copyright, patent, trademark and attribution notices, and prominent notices
of changes in modified files. Its patent and trademark provisions also apply.
Keep `LICENSE_QWEN3.txt`, `NOTICE_QWEN3.txt` and this source record with any
redistributed weights or tokenizer files. OpenNN's software licence does not
replace these model terms.

Neither original model repository has a separate `NOTICE` file at the revisions
above. `NOTICE_QWEN3.txt` is OpenNN's attribution and modification notice,
not a file supplied by Alibaba Cloud.

Credit: Qwen Team, [*Qwen3 Technical Report*](https://arxiv.org/abs/2505.09388),
2025. See the original model cards for the training and evaluation details.

## Converted sources and changes

Artelnics identifies the corresponding original Qwen models in its model
cards and describes a conversion to flat OpenNN binary parameter order with
BF16 and FP32 representations. The conversions inspected are:

- [Artelnics/qwen3-0.6b-opennn at fa990095228d3ba07f32fceba2b25565b48f8922](https://huggingface.co/Artelnics/qwen3-0.6b-opennn/tree/fa990095228d3ba07f32fceba2b25565b48f8922).
- [Artelnics/qwen3-4b-opennn at 5eb7e66549345aa9cae3efe931986fa4e3785322](https://huggingface.co/Artelnics/qwen3-4b-opennn/tree/5eb7e66549345aa9cae3efe931986fa4e3785322).

OpenNN adaptations include parameter ordering and alignment, the binary
representation, tokenizer serialization and a TSV export of special tokens.
The tokenizer reserves `[PAD]` at index zero and shifts original token IDs by
one at load time. The original vocabulary mapping and token text remain intact.

The small model's recorded dimensions are hidden size 1024, 28 layers,
16 query heads, 8 key/value heads, head dimension 128 and intermediate size
3072. The 4B model uses hidden size 2560, 36 layers, 32 query heads,
8 key/value heads, head dimension 128 and intermediate size 9728. Both model
definitions use a vocabulary size of 151936.

The conversion cards describe the BF16 representation as lossless and the
FP32 representation as widening the same values. This records the converter's
description; a complete tensor comparison was not performed during this
licence review. The cards do not pin the exact original revision used for
conversion, so the original revisions above identify the inspected sources
and licence, not a claimed conversion input commit.

### Weight identification

The following sizes and SHA-256 digests come from the pinned Hugging Face
LFS metadata. These full weight files were not downloaded or independently
hashed during this review.

| Conversion | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| 0.6B | `qwen3_bf16.bin` | 1503268864 | `0fc45704fa9c2c621f27bc8b82ba0cbe1005ac68669e810caf9a69577f44c8dd` |
| 0.6B | `qwen3.bin` | 3006537728 | `e86c0566dbe043c24d02179286193e28871bde76eeef7f2e0736102513e49049` |
| 4B | `qwen3_bf16.bin` | 8822858752 | `d56d5631358eabf9e680af310425a3f6ff4dd2032be7ca403d8215abfb4ba34b` |
| 4B | `qwen3.bin` | 17645717504 | `4eb40d9c776cb39412eb3e835c3b97b7630d65a855c3a5bf8a5e724575924f26` |

The factory downloads `qwen3_bf16.bin` from each converted repository.
For the 0.6B variant, its local cache name is `qwen3_draft_bf16.bin`;
for the example's 4B variant, it remains `qwen3_bf16.bin`.

### Tokenizer verification

The tokenizer files were downloaded and checked independently. Both
conversions contain the same tokenizer:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `vocab.json` | 3080770 | `f0f429cf2b3d0870f9b0d12b33a634ab7f4639c2b2b198bd5e9ebb4923749982` |
| `merges.txt` | 1671853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `qwen3_special.tsv` | 548 | `2a90265a3e6deb1254d17f73764ee1b1505be7d7a39c2175aa27818051aca268` |

`merges.txt` matches the original files byte-for-byte. The converted vocabulary
has 151669 entries: the 151643 original base tokens plus the 26 added tokens
from upstream `tokenizer_config.json`. Every original token-to-ID mapping is
preserved; the JSON serialization differs. The TSV records those same
26 special-token IDs and texts.

## Running and redistributing

Build target `qwen3` and run `qwen3 [model_dir]` from the executable directory.
The default cache is `../data/qwen3`, where CMake places the three notices
before the example runs. The default 4B download is about 8.83 GB of BF16
weights plus tokenizer files. Enter a prompt, then `exit` or `quit` to finish.
An empty line also ends the chat.

An existing cache in `../data` can still be used by passing that directory
explicitly. If a different cache is used, keep copies of the three model
notices beside the downloaded files.

Published copies of both Artelnics conversions should include the complete
licence, attribution/modification notice and this source record, and link
to them from their model cards. The model cards currently declare Apache 2.0;
that metadata alone does not provide the complete licence text.
