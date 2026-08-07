# tests/fixtures

Frozen artefacts from older versions of an on-disk format. **Never regenerate
these** — their whole value is that they were written by code that no longer
exists. Regenerating one turns its test from "old rows still load" into "today's
code can read today's code", which is not a claim worth a test.

- `result_row_v1.json` — a `ResultRow` as written before `schema_version` was
  added. `results/` is append-only (CLAUDE.md rule 6), so rows in this shape are
  permanent and must stay loadable forever. Add a new fixture per schema
  version; do not edit this one.
