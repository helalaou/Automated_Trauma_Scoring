# Contributing

Thanks for your interest in this project. It is research code, so the bar for a change is that it
keeps experiments reproducible and results comparable across runs.

## Development setup

1. Clone the repository and run `./install.sh` (creates `.venv/` and installs `requirements.txt`).
2. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
3. Install OpenPose and either put its binary on your `PATH` or set `openpose.binary_path` in
   `config.yaml` (see the [README](README.md#setup)).

## Guidelines

- **Keep changes focused.** One fix or experiment per pull request, with a short description of the
  motivation and what changed.
- **Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org):
  `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`.
- **Experiments are configured, not hard-coded.** Add models, prompt paths and thresholds to
  `config.yaml` rather than to the scripts.
- **Prompts** live in `prompts/` as Jinja2 templates. If you change one, say which task (6-bin or
  3-bin) and models you ran it against, and include the resulting summary numbers.
- **Never commit secrets or outputs.** `.env`, `results/`, `logs/` and `pose_cache/` are ignored on
  purpose. Logs contain full model responses.
- **Data.** Only add images you have the right to redistribute, and never add images of real
  patients. See the dataset notes in the README.

## Reporting bugs and suggesting changes

Use the issue templates. For bugs, include the command you ran, your `config.yaml` changes, the
OpenPose version, and the full traceback.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
