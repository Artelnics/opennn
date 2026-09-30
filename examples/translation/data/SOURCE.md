# Tatoeba Spanish-English sentence pairs

`tatoeba_es_en.tsv` contains 1,000 Spanish-English sentence pairs from
**Tab-delimited Bilingual Sentence Pairs**, compiled by Charles Kelly at
[ManyThings.org](https://www.manythings.org/anki/) from the
[Tatoeba Project](https://tatoeba.org/). The source archive is dated
**2026-02-13**. Each selected row explicitly carries a **CC BY 2.0 France**
notice and the identifiers and usernames of both sentence owners.

The individual contributors are credited in `ATTRIBUTION.tsv`, with a link to
each original sentence. `TATOEBA_README.txt` preserves the archive's original
information, terms and warnings.

## Licence and redistribution

The sentences retain [Creative Commons Attribution 2.0 France](https://creativecommons.org/licenses/by/2.0/fr/deed.en)
([legal text](https://creativecommons.org/licenses/by/2.0/fr/legalcode.fr)).
This licence permits copying, modification and redistribution, including
commercial use, with attribution. The [Tatoeba terms](https://tatoeba.org/en/terms_of_use)
and the ManyThings download page identify the attribution requirement.

When redistributing the dataset, keep `ATTRIBUTION.tsv`, this notice and
`TATOEBA_README.txt` with it; retain the sentence owners' credits, source and
licence links, and describe subsequent modifications. Do not add legal or
technical restrictions that remove the freedoms granted by the licence.
OpenNN's LGPL software licence does not replace these data terms.

## Preparation for OpenNN

Use the original `spa.txt` order and take the first 1,000 rows with distinct
Spanish texts. This gives exactly 1,000 distinct Spanish inputs and keeps the
example small. Repeated English targets are allowed. The demonstration phrase
`Vete.` occurs in this subset with the target `Go.`.

Reverse the original English/Spanish columns, keeping the sentence text,
punctuation and accents unchanged. Write a UTF-8, two-column, tab-delimited
file with CSV quote escaping, no header and LF line endings. Keep the original
attribution field in a separate TSV so it does not become model input or a
translation target. `ATTRIBUTION.tsv` maps every data row to its source row,
English and Spanish identifiers, owners and original attribution statement.

The example creates its own random training/validation/testing split. This
subset is for demonstrating encoder-decoder training, rather than measuring
general translation quality.

Source archive:
[spa-eng.zip](https://www.manythings.org/anki/spa-eng.zip),
SHA-256 `07bb8aaaf1458abbcdc8a3a51ce5b9f8a2643c829dcb7c2a63b3fec8dde934dc`.
The download URL can be updated by its publisher; regeneration checks this
hash and requires the recorded version if the live download changes.

Bundled file SHA-256:

- `tatoeba_es_en.tsv`: `bd2290e815cf2ed185aa378532edd6048fa85795c66b4e757da581610e291e46`.
- `ATTRIBUTION.tsv`: `05b20f703fc44fae7d0f7f3373009aaa470baa9b6ce68e808dea161ebcfc21c8`.
- `TATOEBA_README.txt`: `262eaba34251ebb2e7c5d514c5a2b856fe6eec058cee957d22a6c6129947a228`.

## Reproduce

Run this with Python 3.12 from the repository root. It uses only the standard
library, puts downloads and output outside the checkout, and verifies the
source before generating the dataset and its attribution notice. No Python
script needs to be saved.

```python
import csv
import hashlib
from pathlib import Path
import re
from urllib.request import Request, urlopen
import zipfile

CACHE = Path("../opennn-tatoeba-source")
OUT = Path("../opennn-tatoeba-regenerated")
URL = "https://www.manythings.org/anki/spa-eng.zip"
EXPECTED = "07bb8aaaf1458abbcdc8a3a51ce5b9f8a2643c829dcb7c2a63b3fec8dde934dc"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
archive = CACHE / "spa-eng.zip"
if not archive.exists():
    request = Request(URL, headers={
        "User-Agent": "OpenNN-example-data/1.0",
        "Accept": "application/zip, */*;q=0.8",
    })
    with urlopen(request, timeout=60) as response:
        archive.write_bytes(response.read())
if hashlib.sha256(archive.read_bytes()).hexdigest() != EXPECTED:
    raise ValueError("Source archive differs from the recorded 2026-02-13 release")

with zipfile.ZipFile(archive) as source:
    rows = [(i, *line.split("\t")) for i, line in enumerate(
        source.read("spa.txt").decode("utf-8").splitlines(), 1)]
    about = source.read("_about.txt")
assert len(rows) == 144215
assert all(len(row) == 4 for row in rows)
selected = []
seen = set()
for row in rows:
    if row[2] not in seen:
        selected.append(row)
        seen.add(row[2])
        if len(selected) == 1000:
            break
assert len(selected) == len(seen) == 1000
pattern = re.compile(
    r"CC-BY 2\.0 \(France\) Attribution: tatoeba\.org "
    r"#(\d+) \(([^)]+)\) & #(\d+) \(([^)]+)\)")

with (OUT / "tatoeba_es_en.tsv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerows((es, en) for source_row, en, es, credit in selected)

with (OUT / "ATTRIBUTION.tsv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(("dataset_row", "source_row", "spanish_id", "spanish_owner",
                     "spanish_url", "english_id", "english_owner", "english_url",
                     "original_attribution", "licence_url"))
    for dataset_row, (source_row, en, es, credit) in enumerate(selected, 1):
        match = pattern.fullmatch(credit)
        if match is None:
            raise ValueError(f"Missing recorded licence or attribution: row {source_row}")
        en_id, en_owner, es_id, es_owner = match.groups()
        writer.writerow((dataset_row, source_row, es_id, es_owner,
            f"https://tatoeba.org/en/sentences/show/{es_id}", en_id, en_owner,
            f"https://tatoeba.org/en/sentences/show/{en_id}", credit,
            "https://creativecommons.org/licenses/by/2.0/fr/"))
(OUT / "TATOEBA_README.txt").write_bytes(about)
for name in ("tatoeba_es_en.tsv", "ATTRIBUTION.tsv", "TATOEBA_README.txt"):
    print(name, hashlib.sha256((OUT / name).read_bytes()).hexdigest())
```
