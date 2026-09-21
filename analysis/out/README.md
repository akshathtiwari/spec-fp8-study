# Derived tables

**Generated, not authored.** Every file here is produced by
`analysis/build_tables.py` from `results/`, and carries a header naming the
script, its inputs, and the code SHA that produced it.

Do not edit these by hand. If a number is wrong, the script is wrong — fix the
script and regenerate, so the error cannot survive in one place while being
fixed in another.

```bash
python analysis/build_tables.py
```

| table | what |
|---|---|
| `runs.md` | every run, its GPU, engine version, cell count and GPU-seconds |
| `compatibility_matrix.md` | latest record per cell: status, tau, acceptance, KV capacity |
| `backend_attribution.md` | which attention backend actually served each cell, read from its log |
| `correctness_tests.md` | Tests 1 and 2 over matched pairs |
| `kv_capacity.md` | measured FP8-KV capacity gain, backend held fixed |
