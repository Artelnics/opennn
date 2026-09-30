# GoEmotions data

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., Ravi, S.
(2020), *GoEmotions: A Dataset of Fine-Grained Emotions*, ACL 2020, pp. 4040-4054.
[Paper and citation](https://aclanthology.org/2020.acl-main.372/),
DOI [10.18653/v1/2020.acl-main.372](https://doi.org/10.18653/v1/2020.acl-main.372).
Published by Google Research; the comments come from Reddit.

Data licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Google Research explicitly licenses its datasets under CC BY 4.0 in its
[repository README](https://github.com/google-research/google-research/blob/d36068b845da4c2b24927fee2cea1e6ef98dadda/README.md).
The Apache 2.0 licence in that repository applies to its source code.
Preserve this attribution, source and licence link, and the description of
changes below when redistributing the data. Do not imply endorsement by Google
or the authors. Licence checked on 2026-09-30.

## Source and OpenNN adaptation

Source revision: `d36068b845da4c2b24927fee2cea1e6ef98dadda`.
Original [dataset description](https://github.com/google-research/google-research/blob/d36068b845da4c2b24927fee2cea1e6ef98dadda/goemotions/README.md),
[train.tsv](https://raw.githubusercontent.com/google-research/google-research/d36068b845da4c2b24927fee2cea1e6ef98dadda/goemotions/data/train.tsv)
and [emotions.txt](https://raw.githubusercontent.com/google-research/google-research/d36068b845da4c2b24927fee2cea1e6ef98dadda/goemotions/data/emotions.txt).

`goemotions.txt` contains all 5,272 single-label records for `anger`, `fear`,
`joy`, `love`, `sadness` and `surprise` from the original 43,410-row training
file, in source order. Multi-label records and all other categories are omitted;
the original development and test files are not used. Numeric labels are mapped
through `emotions.txt`, and comment identifiers are removed. Text is preserved,
with TSV quoting to escape double quotes. There is no header: each row has text
and an emotion label separated by a tab.

| Emotion | Messages |
| --- | ---: |
| anger | 1,025 |
| fear | 430 |
| joy | 853 |
| love | 1,427 |
| sadness | 817 |
| surprise | 720 |

The example creates its own stratified 80/10/10 split. Its results are for this
six-class subset and are not the published GoEmotions benchmark results.
The original publisher notes biases from Reddit and the annotation process,
and potentially problematic content; see the dataset description linked above.

## Reproduce the bundled file

Run this Python 3.12 snippet from the repository root. It downloads the two
pinned files, checks the original training file's SHA-256, and writes the
bundled subset. This requires internet access.

```python
import csv
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen

revision = "d36068b845da4c2b24927fee2cea1e6ef98dadda"
base = f"https://raw.githubusercontent.com/google-research/google-research/{revision}"
with urlopen(f"{base}/goemotions/data/train.tsv", timeout=60) as response:
    source = response.read()
if hashlib.sha256(source).hexdigest() != "1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5":
    raise ValueError("GoEmotions train.tsv differs from the reviewed source")
with urlopen(f"{base}/goemotions/data/emotions.txt", timeout=60) as response:
    labels = response.read().decode("utf-8").splitlines()

selected = {"anger", "fear", "joy", "love", "sadness", "surprise"}
output = io.StringIO(newline="")
writer = csv.writer(output, delimiter="\t", lineterminator="\n")
for line in source.decode("utf-8").splitlines():
    text, identifiers, _ = line.split("\t")
    if "," not in identifiers and labels[int(identifiers)] in selected:
        writer.writerow((text, labels[int(identifiers)]))
Path("examples/emotion_analysis/data/goemotions.txt").write_bytes(
    output.getvalue().encode("utf-8"))
```

SHA-256 of original `train.tsv`:
`1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5`.

SHA-256 of bundled `goemotions.txt` (UTF-8, LF line endings):
`19e0a3169d688131af206bfbad88501fcb18c3153d815e7584fabb2c795e35cf`.
