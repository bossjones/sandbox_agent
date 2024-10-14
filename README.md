# sandbox_agent

sandbox to play around w/ different agent/rag techniques


To install dependencies:
```bash
rye sync --all-features
```

To run the agent:

```bash
rye run sandboxctl
```

## Command-line Arguments

The `sandboxctl` command supports several subcommands and options:

- `version`: Print the version of sandbox_agent.
  ```bash
  rye run sandboxctl version
  ```

- `deps`: Display version information for sandbox_agent and its dependencies.
  ```bash
  rye run sandboxctl deps
  ```

- `about`: Display information about the GoobBot CLI.
  ```bash
  rye run sandboxctl about
  ```

- `show`: Show sandbox_agent information.
  ```bash
  rye run sandboxctl show
  ```

- `run-pyright`: Generate type stubs for GoobAI.
  ```bash
  rye run sandboxctl run-pyright
  ```

- `go`: Start the GoobAI Bot.
  ```bash
  rye run sandboxctl go
  ```

For more detailed information on each command, you can use the `--help` option:

```bash
rye run sandboxctl --help
```

This was created using [rye](https://rye.astral.sh/guide/getting-started/introduction/).
