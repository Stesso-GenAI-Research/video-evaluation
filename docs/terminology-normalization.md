# Terminology normalization: audit and implementation direction

Audited September 22, 2026. Source annotations are authentic human-generated
data. Authorship and terminology consistency are separate properties.

## Findings from the complete local dataset

The audit reads all 250 source videos / 1,703 raw annotations and all 1,663
canonical clips. Counts below use the canonical index unless explicitly marked
raw. Input SHA-256 hashes are in the generated `summary.json`.

| Finding | Count | Interpretation |
|---|---:|---|
| Parsed tool item occurrences / distinct exact names | 3,222 / 1,931 | Names include brands, attributes and parser fragments |
| Parsed supply item occurrences / distinct exact names | 874 / 704 | Distinct names are not distinct concepts |
| Tool / supply names occurring once | 1,447 / 601 | Review by frequency plus context; rare does not mean wrong |
| Inventory occurrences with unbalanced parentheses | 111 | 95 tool + 16 supply; parser or source review required |
| Inventory occurrences with placeholder text in names | 91 | 25 tool + 66 supply; flags can overlap other categories |
| Inventory occurrences with alternatives | 490 | 385 tool + 105 supply; alternatives are not necessarily synonyms |
| Clips without description | 1,081 (65.0%) | Weakens contextual disambiguation |
| Clips without goal | 1,188 (71.4%) | End states cannot be reliably recovered from a synonym table |

The fields also mix roles. For example, the parsed tools include `Screws`
(14 occurrences) and `Shower Enclosure`; these can be a fastener/material or
the object acted on, rather than the instrument used. This needs a separate
role review instead of correction solely by word similarity.

### Observed terminology variants

Counts are numbers of clips with a case-insensitive, whole-phrase mention in
clip name, description, goal, source tools or source supplies. They include
mentions in alternatives and purpose text, exclude parent-video narrative,
and can overlap within a group. They are not counts of tools actually used.

| Variants | Clip counts | Treatment |
|---|---|---|
| drywall / gypsum board / plasterboard | 21 / 1 / 3 | Shared board concept after contextual review |
| measuring tape / tape measure | 32 / 33 | Shared measuring-tool concept; preserve subtype |
| crescent wrench / adjustable wrench | 4 / 11 | Candidate alias; retain brand, do not merge pipe wrenches |
| shop vac / shop vacuum | 1 / 5 | Candidate alias; retain wet/dry capabilities |
| drywall mud / joint compound | 2 / 6 | Drywall-finishing context only |
| Sheetrock | 2 | These clips name joint compound, not board |

`Spanner`, `mitre saw` and `hex key` have no whole-phrase mentions in the audited
fields; `miter saw` occurs in 19 clips and `Allen key` in 2. Regional aliases
can still matter for future queries, but this dataset does not measure their
frequency in real user requests.

Examples with traceable evidence:

- `indexed-video-125591-segment-19p618-34p259` explicitly lists drywall with
  plasterboard and gypsum board as alternatives. This group is also supported
  by the [Gypsum Association](https://gypsum.org/what-is-gypsum-board/), which
  identifies drywall as a common name for gypsum board.
- `indexed-video-273984-segment-148p4-176p88` names Sheetrock Easy Sand 20 Joint
  Compound. A brand-name replacement would change the material identity.
  [USG's joint treatment guide](https://assemblies-tools.usg.com/content/usgcom/en/blog/usg-joint-treatment-selection-guide.html)
  confirms Easy Sand is a joint-compound product.
- `indexed-video-3400303-segment-24p083-51p65` contains a complete parenthesized
  safety-gear list in the source; the inventory parser splits at its internal
  commas, creating fragments. This is a pipeline parsing problem, not evidence
  that the human author wrote broken fragments.
- `indexed-video-34228-segment-221p53-236p24` has the goal `Verify Watertight
  Seal`. Verification, achieving a seal and observing a successful seal must
  remain distinguishable.

## What the current system misses

`normalize_term` cleans case, punctuation and spacing. It does not map synonyms
to concepts. TF-IDF uses word unigrams and bigrams without a terminology
lexicon. Structured matching uses lemma overlap for objects/tools/supplies;
VerbNet and FrameNet provide action backoffs, not a domain noun lexicon.

Inventory alternatives are included in retrieval context, although they may
name substitutes rather than the instrument actually used. The parser's
comma-splitting behavior also needs correction before vocabulary counts can
be treated as a clean concept inventory.

These findings establish terminology and parsing weaknesses. They do not
establish how much of the previously observed performance gap they explain.

## Target representation

Keep original text and source references. Add a versioned concept layer:

| Field | Purpose |
|---|---|
| `concept_id`, preferred label | Stable identity independent of wording |
| aliases, locale, domain/sense | Regional names with contextual restrictions |
| relation | Exact alias, subtype, brand/product, substitute, or related concept |
| attributes | Size, units, material, rating, grit, wet/dry capability, etc. |
| role | Tool, supply, acted-on object, protective equipment, or other |
| source span and evidence | Trace every mapping back to the annotation |
| review status and lexicon version | Reproduce and revise mappings |

For actions, preserve object and direction: removing a faucet and installing
one have opposite results. For end states, separately represent initial state,
action, intended final state and explicitly observed final state. Missing
states remain unknown. `Smooth`, `flat`, `level`, `clean`, `dry` and `cured`
must not collapse into one generic completion state. A stated goal does not
prove that the clip achieves it.

The seed JSON contains 12 proposed groups with scope restrictions and explicit
exclusions. It is a review artifact, not an activated synonym dictionary.
In particular, broad relations and substitutes must not become exact aliases.

## Tools and vocabulary sources

- **OpenRefine:** use the local inventory CSV for facets and spelling/format
  clustering. Its [clustering documentation](https://openrefine.org/docs/technical-reference/clustering-in-depth)
  explains that clustering is syntactic, so it will not itself solve gypsum
  board versus drywall. Its [reconciliation workflow](https://openrefine.org/docs/manual/reconciling)
  can propose external concepts for review.
- **Getty AAT:** investigate architecture/material/technique concepts and
  scope notes as a seed, then measure actual dataset coverage. It does not
  cover every domain in this 26-category sample. See the
  [vocabulary overview](https://www.getty.edu/research/tools/vocabularies/) and
  [data access and attribution terms](https://www.getty.edu/research/tools/vocabularies/obtain/).
- **WordNet:** use sense-specific synonyms as proposals, not indiscriminate
  expansion. [Princeton](https://wordnet.princeton.edu/) describes the sense
  structure and notes that its own database is no longer developed, pointing
  to community successors. The project already installs NLTK WordNet resources.
- **spaCy PhraseMatcher:** implement reviewed multiword alias matching using
  the existing dependency. It supports terminology lists and case-insensitive
  matching; context checks and longest-match handling still need project code.
  See [official API](https://spacy.io/api/phrasematcher).

Recommendation: a small project-owned lexicon informed by these sources,
reviewed in context and applied deterministically. Similarity tools or an LLM
may propose candidates, but their suggestions do not establish equivalence.

## Next experiment

1. Repair inventory splitting while preserving parentheses, units, brand,
   purpose and alternative relations; add regression cases from the audit.
2. Review the seed aliases and high-frequency inventory terms against their
   source clips. Keep ambiguous cases unresolved. Expand by category.
3. Add optional concept features to both query and candidate processing,
   preserving raw text. Keep the recorded frozen baseline intact; normalization
   is a separately versioned challenger.
4. Compare raw baseline, parser cleanup only, and parser cleanup plus lexicon
   on identical queries/candidates. Add human-checked synonym and regional
   paraphrases, plus confusing non-synonyms such as compound versus board,
   pipe wrench versus adjustable wrench, and remove versus install.
5. Split by source video before development; reserve unseen aliases/videos for
   the final check. Audit new equivalences for benchmark query/target leakage
   and report that separately rather than silently changing eligibility.
6. Measure mapping precision, unresolved coverage, false merges, Hit@1/3/10,
   MRR and clustered differences. Report synonym-query improvements alongside
   ordinary-query regressions. Do not derive new preference winners from the
   normalizer being evaluated.

End-state extraction should be evaluated separately against reviewed state
labels; it is a larger task than noun terminology normalization.

## Reproduce and review

From the repository root, with the existing canonical index:

```bash
python3 scripts/audit_terminology.py
```

The script uses only Python's standard library. It writes to
`project1_outputs/terminology-audit/`:

- `summary.json`: counts, input hashes and limitations.
- `inventory.csv`: all exact parsed names, frequencies and example clip IDs.
- `inventory_review.csv`: parser/placeholder flags and raw item fragments.
- `alias_counts.csv`: observed clip counts for each proposed alias.
- `alias_review.csv`: every matching field, full context, clip ID and blank
  review decision column. Re-running overwrites these audit outputs; use a new
  `--output` directory after entering review decisions.

The audit and seed do not change the source annotations or retrieval scores.
