# How the system works

This page explains the code at the level of a first undergraduate data
structures or information retrieval course. The important idea is that the
pipeline turns a nested video export into searchable records, then ranks those
records in three different ways.

## The full pipeline

```text
indexed-videos-*.jsonl
          |
          v
validate and merge clip annotations
          |
          v
canonical timestamped clips
          |
          v
extract actions, objects, tools, and supplies
          |
          v
lexical / structured / hybrid search
          |
          v
automatic benchmark or blinded human comparison
```

Each arrow is a repeatable program step. The build also records hashes and
version information so later commands can tell whether an index is stale.

## 1. Reading the nested video file

Each line of `indexed-videos-250.jsonl` describes one parent video. A video
contains a nested `clips` list, and each clip annotation describes a time range
inside that video. The parser walks all the way into those lists; it does not
treat one line as one searchable clip.

The build performs three important cleanup steps:

1. It rejects a clip when its end time is not later than its start time and
   writes the row to `input/rejected_clips.jsonl`.
2. It groups rows that point to the same video and timestamp range.
3. It merges the useful metadata from repeated rows instead of silently
   throwing it away. Alternate titles become aliases and the original rows stay
   available as provenance.

The result is a set of distinct, playable segments. A canonical clip ID is
built from the source video ID and timestamps. This makes the ID stable even if
the input lines are reordered. The current sample audit is recorded in the
[project status guide](stesso-project-update.md#31-real-indexed-video-sample).

The parser retains both clip-level and parent-video fields. These include names,
descriptions, goals, categories, source information, engagement counts, URLs,
YouTube IDs, tools, and supplies. Tool and supply text is split into item names,
alternatives, and stated purposes, but the original text is kept as well.

## 2. Extracting action semantics

The next stage reads the natural-language fields and looks for records like:

```text
sentence: Tighten the set screw with a screwdriver.
action:   tighten
object:   screw
tool:     screwdriver
```

spaCy supplies tokens, parts of speech, and dependency relationships. The code
uses those relationships to connect a verb to its object and directly attached
tool phrase. It also keeps the clip's parsed tool and supply inventories as
additional context.

This distinction matters. A screwdriver listed for a video is evidence that the
clip may use a screwdriver; it does not prove that the grammar attached that
tool to every action in the video. The output records these two kinds of
evidence separately.

VerbNet and FrameNet provide broader semantic classes when two action words are
different but related. For example, a shared verb class may provide weaker
evidence after an exact action match fails. The build also creates an early DIY
action taxonomy. That taxonomy is exploratory and has not been manually
validated, so production ranking does not use it.

Current extraction coverage is summarized in the
[project status guide](stesso-project-update.md#31-real-indexed-video-sample).
Coverage means the parser found a field; it is not the same as human-checked
precision.

## 3. Building three search methods

All methods search the same canonical clips and return the same kind of result
record. Only the scoring rule changes.

### Lexical search

Lexical search uses TF-IDF. In simple terms, it gives weight to query words that
are important in a clip and relatively uncommon across the collection. A clip
with words such as `remove` and `faucet` receives a high score for `remove old
faucet`.

This is the baseline because it is simple, fast, and often strong. It can still
confuse related language when the exact action direction matters.

### Structured search

Structured search first parses the query. It then compares one complete query
action record with one complete candidate action record. The current score uses:

| Component | Normal weight | Question it asks |
|---|---:|---|
| Action | 55% | Are the actions compatible? |
| Object | 35% | Do they act on the same thing? |
| Tool or supply context | 10% | Do they use the requested item? |

If a query does not name an object or context item, the unused weight is divided
among the components that are present. This avoids punishing a short query just
because it has no tool.

Several guardrails keep the score interpretable:

- Action, object, and context must come from aligned action records. The scorer
  cannot take a verb from one sentence and an unrelated object from another.
- Object and context evidence only helps when the actions are compatible.
- A positive action and its negated form do not count as a match.
- A phrase such as `with primer` is compared with both tool and supply
  inventories because a general dependency parser may not know that primer is
  consumed while a screwdriver is a tool.
- Location phrases such as `on the wall` are reported as diagnostics, but do
  not increase the production score.
- A clip must receive a positive score to be returned. The program does not add
  unrelated zero-score clips just to fill the requested list length.

Short commands such as `paint wall` do not always look like full sentences to
the parser. The code retries known terse verb forms as commands. If parsing
still fails, hybrid search falls back to lexical results and records a warning.

### Hybrid search

Hybrid search combines the two normalized scores:

```text
hybrid score = alpha * lexical score + (1 - alpha) * structured score
```

The current default is `alpha = 0.5`, so both sides receive equal weight. This
is a fixed experimental setting, not a proven optimum. A future tuning study
must choose the weight on development videos and then evaluate it once on
separate held-out videos.

## 4. Returning useful evidence

A search result includes its canonical clip ID, source video ID, title, URL,
start and end times, total score, and component signals. Those component values
make it possible to ask why a result appeared. For example, a result may have
an exact action and object match but no matching tool.

The CLI can return any top-*k* value and can limit how many clips come from one
source video. The normal search code also writes a JSON report, which makes a
run reproducible and easier to inspect than terminal text alone.

## 5. Evaluating the result lists

There are three separate evaluation paths:

- The **automatic development benchmark** uses fields already paired inside the
  sample. It runs without new labels and is useful for finding weaknesses.
- The **pairwise preference evaluation** directly scores an already-judged A/B
  pair for one project step. It is the frozen Project 1, Step 1 experiment.
- The **old-versus-new human comparison** takes the real old top results,
  generates new results, hides the system names, and asks a person which clips
  are relevant. This is the experiment needed for the main quality claim.

All three paths are implemented. The repository does not yet contain the
authentic W25 pairwise judgments or the supervisor's historical rankings and
completed review. See
[Experiment methodology](experiments-and-results.md) for the distinctions and
the [project status guide](stesso-project-update.md) for current results.

## What this system is not

The repository is a research CLI and batch pipeline. It is not yet a web UI,
deployed API, persistent search server, or connection to the private production
database. It searches annotation text in the supplied export; the sample does
not include video frames, captions, transcripts, or dense embedding vectors.
