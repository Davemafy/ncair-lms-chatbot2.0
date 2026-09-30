# Contributing

Use a focused branch and keep changes scoped to one concern.

Before opening a PR:

```bash
make check
```

For Python changes, `ruff` owns formatting and import order. Unit tests must stay offline and must not download N-ATLaS, call Ollama, or require Cloudflare.

The benchmark is held-out evaluation data. Do not copy benchmark questions into router prompts, tool descriptions, examples, or training fixtures. If the benchmark changes, run `make benchmark-validate` and explain why the change is necessary.

Integration tests that load the real N-ATLaS model are opt-in:

```bash
make test-integration
```
