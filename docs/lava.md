# Orchestration integration (`varscore.lava`)

varscore scores variants. It does not schedule work, move files between
machines, or decide what runs where. A platform built on `lava-core` does that,
and to drive varscore it needs a description of two things:

- **how to run varscore** — which container commands, in what order, reading and
  writing which paths;
- **how to read what came out** — which score columns exist, and what makes a
  variant worth prioritizing.

`varscore/lava/` is that description, as a `ModelPlugin`.

## Why it lives here

Every fact in the plugin is a fact about varscore: the command modules and their
flags, the fold count the CLI enforces, the prioritization thresholds. When the
plugin lived on the platform side, those facts sat in a different repository from
the code implementing them, written as string literals, with nothing checking the
two agreed. Renaming a flag in `score.py` broke the caller silently, and the
failure surfaced as an argparse usage error inside a GPU container, minutes into
a job.

Stated here, they can be tested against the code they describe.

## The two halves

The subpackage is split by what it depends on, and the split is load-bearing:

| Module | Imports `lava-core`? | Tested by |
| --- | --- | --- |
| `varscore.lava.commands` — the argv contract | **No** | `tests/test_lava_commands.py`, in every install |
| `varscore.lava.chrombpnet` — the plugin | Yes | `tests/test_lava_plugin.py`, needs `lava-core` |

`lava-core` is private, so tests that need it cannot run for fork PRs and are a
best-effort gate. Keeping the argv contract framework-free means the check that
actually catches day-to-day drift — a renamed flag — runs unconditionally in
ordinary CI, on every supported Python version, with no credentials.

Importing `varscore.lava` does not pull in `lava-core`: `ChromBPNetPlugin` is
resolved lazily through a module `__getattr__`, so `from varscore.lava import
commands` works in a base install. Don't make that import eager.

## Installing

```bash
# Python 3.12 only -- see below.
pip install -r requirements-lava.txt
```

`lava-core` lives in a private repository, so this resolves only with read access
to it; without access pip fails with an authentication error. It is not published
to any public index.

**Python 3.12 is the only supported version for this integration.** `lava-core`
requires >=3.12 and varscore supports <3.13, so 3.12 is the whole overlap.

### Why it is not a `[lava]` extra

Declaring `lava-core` in `pyproject.toml` would break the resolver for everyone,
including contributors who never touch this integration. It is a *source*
dependency requiring >=3.12, and uv builds a source dependency's metadata using
an interpreter valid for varscore's own floor (3.9) — so `uv lock` becomes
unsatisfiable on every Python version, not just old ones. Being private, it would
also fail outright for anyone without repository access, which for a public repo
means anyone.

A separate requirements file keeps it out of the resolver's view unless asked
for.

## Using it

Installing is normally all that is needed. `ChromBPNetPlugin` is advertised
through the `lava.model_plugins` entry point (declared in `pyproject.toml`), and
the platform discovers it on its own.

```python
from lava_core.plugins.model.plugin import get_model_plugin

plugin = get_model_plugin("CHROMBPNET")
```

Call `varscore.lava.register()` only to break a tie. The registry merges every
entry point for a group into one mapping keyed by architecture name, so if
another installed distribution also advertises `CHROMBPNET`, which one wins comes
down to discovery order. Programmatic registration is applied after entry points
and overrides them:

```python
import varscore.lava

varscore.lava.register()  # varscore's plugin now wins deterministically
```

This matters today: `lava-core` still ships its own `CHROMBPNET` plugin, and with
both installed **`lava-core`'s wins**. See "Migration status" below.

## The argv contract

`varscore/lava/commands.py` is the interface between varscore and any caller that
runs it in a container. The scoring image's entrypoint is `python -m`, so a task
is `python -m varscore.scoring.chrombpnet.score -m ... -p ... -g ...`.

| Builder | Command module | Role |
| --- | --- | --- |
| `score_argv` | `varscore.scoring.chrombpnet.score` | Score variants through one fold |
| `predictions_argv` | `varscore.scoring.chrombpnet.predictions` | Average folds into model-level scores |
| `ingest_argv` | `varscore.scoring.chrombpnet.ingest` | Derive peak distributions + interval tree |
| `interpretation_argv` | `...interpret.interpretation` | Contribution scores for one fold |
| `average_interpretations_argv` | `...interpret.average_interpretations` | Average fold contributions |
| `prepare_variant_plotting_argv` | `...interpret.prepare_variant_plotting` | Render per-variant plots |

**Build argv with the builders, never by hand.** They are the only place flag
names are written down, so the plugin cannot drift from the parser
independently — and the round-trip test (build with the builder, parse with the
real parser) catches a rename with nothing extra installed.

Motif hit-calling is *not* in this table: it runs finemo, a separate tool in its
own image.

```python
from varscore.lava import commands

argv = commands.score_argv(model="/m.h5", peak_distribution="/pd.npy",
                           genome="/g.fa", variants="/v.tsv", out="/o.tsv")
commands.validate_argv(argv)                                  # one command
commands.validate_all(task.command for task in plan.shards)   # a whole plan
```

Both raise `ArgvContractError` on a mismatch, carrying argparse's own diagnostic.

### Command ownership is by prefix

`validate_all` skips commands belonging to other tools, and decides by the
`varscore.` module prefix — *not* by membership of `COMMANDS`. A membership test
would answer "not ours" for a **misspelled** varscore module and skip right over
it, passing on exactly the mistake this contract exists to catch. Anything under
`varscore.` is claimed and must validate.

### The fold count is structural

`commands.NUM_FOLDS` is 5, and that is a constraint rather than a default.
`ingest`, `predictions`, and `average_interpretations` each declare exactly five
`required=True` fold flags (`-f0`…`-f4`). A model scored with any other fold
count cannot be passed to them — there is no flag to put a sixth fold on, and
omitting one of the five fails as a missing required argument. The builders
reject a wrong-sized fold list rather than emitting argv the command would
refuse.

## The promoter rule

varscore states the ChromBPNet prioritization rule twice: as a pandas expression
in `varscore/prioritization.py`, for scoring in-process, and as a predicate in
`varscore/lava/chrombpnet.py`, for a platform to push into SQL.

Both test the **`in_promoter` membership flag**. Neither may test
`region_type == 'promoter'`.

A variant overlaps a *set* of regions; `region_type` is only the
severity-collapsed headline. A variant that is both exonic and in a promoter
reports `region_type = 'exonic'`, so testing the collapsed label drops its
promoter membership and fails to prioritize it. This is the general rule in
[region classification](region_classification.md) — test membership, never
`region_type == "x"` — and it has caused a real prioritization bug.

The two implementations disagreed on exactly this until the plugin moved here,
which is the clearest argument for the move: the rule diverged precisely because
it was stated on both sides of a repository boundary. `TestPrioritizationRule`
checks the behaviour on the disputed row (`region_type="exonic"`,
`in_promoter=True`) and cross-checks the two implementations against each other.

`in_promoter` is a boolean the annotation source projects alongside
`region_type`, so the predicate needs nothing new to read it.

## Migration status

`lava-core` currently ships its own copy of this plugin, also registered for
`CHROMBPNET`. Both are discoverable when both are installed, and the winner is
discovery order.

`TestDropInParity` keeps that harmless by requiring the two to be
indistinguishable. It compares **whole plan objects** for all three plan types,
not just argv — labels, resources, and declared transfers are observable
orchestration behaviour too, since a shard that requests the wrong GPU pool or
omits an input transfer is a different shard however identical its command line.

To complete the move: delete `lava-core`'s `CHROMBPNET` entry point and its
`lava_core/plugins/model/chrombpnet.py`, then delete `TestDropInParity` here — it
exists only to compare against a copy that will no longer exist.
