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

Stated here, they can be tested against the code they describe — which is what
`tests/test_lava_plugin.py` does.

## Installing

The integration is optional and isolated. It is the only part of varscore that
imports `lava-core`, nothing in the scoring path imports it, and the scoring
container does not install it.

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

| Name | Command module | Role |
| --- | --- | --- |
| `SCORE` | `varscore.scoring.chrombpnet.score` | Score variants through one fold |
| `PREDICTIONS` | `varscore.scoring.chrombpnet.predictions` | Average folds into model-level scores |
| `INGEST` | `varscore.scoring.chrombpnet.ingest` | Derive peak distributions + interval tree |
| `INTERPRETATION` | `...interpret.interpretation` | Contribution scores for one fold |
| `AVERAGE_INTERPRETATIONS` | `...interpret.average_interpretations` | Average fold contributions |
| `PREPARE_VARIANT_PLOTTING` | `...interpret.prepare_variant_plotting` | Render per-variant plots |

Motif hit-calling is *not* in this table: it runs finemo, a separate tool in its
own image. Use `is_varscore_command` to filter a mixed plan.

Validate argv against the real parsers:

```python
from varscore.lava import commands

commands.validate_argv([commands.SCORE, "-m", "model.h5", ...])   # one command
commands.validate_all(task.command for task in plan.shards)       # a whole plan
```

Both raise `ArgvContractError` on a mismatch, carrying argparse's own diagnostic.
Every command module exposes `build_parser()` and imports without the `[model]`
extra, so this works in a base install with no GPU and no TensorFlow.

### The fold count is structural

`commands.NUM_FOLDS` is 5, and that is a constraint rather than a default.
`ingest`, `predictions`, and `average_interpretations` each declare exactly five
`required=True` fold flags (`-f0`…`-f4`). A model scored with any other fold
count cannot be passed to them — there is no flag to put a sixth fold on, and
omitting one of the five fails as a missing required argument. A caller that
varies its fold count independently emits argv these commands reject.

## Known divergence: `region_type` vs `in_promoter`

varscore states the ChromBPNet prioritization rule **twice**:

- `varscore/prioritization.py` — a pandas expression, for scoring in-process.
- `varscore/lava/chrombpnet.py` — a predicate, for a platform to push into SQL.

They should agree. They do not. The pandas version tests the `in_promoter`
membership flag; the predicate tests `region_type == 'promoter'`.

`region_type` is a severity-collapsed *single* label, so a variant that is both
exonic and in a promoter reports `region_type == 'exonic'`. It fails the
predicate's promoter test while passing the pandas one. This is precisely the
failure mode this repo's conventions warn about — test membership, never
`region_type == "x"` — and it has caused a real prioritization bug before.

The predicate is deliberately left as-is so that moving the plugin changes
nothing observable. Fixing it changes prioritization output and is a separate,
deliberate decision.

Both sides are pinned by `TestKnownDivergence`, which fails the moment either
changes. When they are reconciled, delete that test rather than updating it. The
cheap fix is a boolean `in_promoter` annotation column, since the predicate DSL
has comparison operators but no membership operator.

## Migration status

`lava-core` currently ships its own copy of this plugin, also registered for
`CHROMBPNET`. Both are discoverable when both are installed, and the winner is
discovery order.

`TestDropInParity` keeps that harmless by requiring the two to be
indistinguishable: same model type, fold count, score columns, predicate SQL,
per-model score entries, shard kinds, image names, and byte-identical argv for
all three plan types.

To complete the move, delete `lava-core`'s `CHROMBPNET` entry point and its
`lava_core/plugins/model/chrombpnet.py`, then delete `TestDropInParity` here —
it exists only to compare against a copy that will no longer exist.
