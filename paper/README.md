# Manuscript draft

`report.pdf` is the compiled single-column manuscript. `main.tex` is the editable
source; `references.bib` contains the verified references. The latest draft is
dated October 15, 2026 and reports the October 6 experiments. Author and
affiliation remain placeholders.

Build from the repository root with TeX Live/MacTeX and `latexmk` installed:

```bash
bash paper/build.sh
```

The script verifies the recorded result hashes, regenerates numerical tables
from the completed experiment suite, runs LaTeX/BibTeX, and writes
`paper/report.pdf` and the portable LaTeX bundle `paper/source.zip`.
To use a different completed suite, pass its absolute path
as the first argument. The prose describes the October 6 run and must also be
reviewed if using new results; regenerating tables does not update every claim.

To build without the private local experiment files, use the included generated
tables: run `latexmk -pdf -outdir=build main.tex` from this directory. The source
bundle includes those tables and the bibliography; it excludes source datasets
and human-review worksheets.

The manuscript leaves explicit placeholders for related work, the original
annotation procedure, independent human preference evaluation, reviewed parser
accuracy, prospective held-out results, acknowledgments, and final conclusions.
It reports a negative development result without presenting constructed labels
as human preferences. The protocol and study tools are described in
`../docs/study-completion.md`.

Before submission, supply the author information, finalize the literature
review, verify permissions for private data, complete the human studies, and
replace the marked result placeholders. The current draft is not a completed
validation study.
