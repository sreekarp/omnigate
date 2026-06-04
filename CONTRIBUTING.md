# Contributing to OmniGate

Thanks for your interest in improving OmniGate! This guide covers local setup,
the conventions we follow, and how to add common things (a provider, a
migration, a test). By contributing you agree your work is licensed under the
project's [MIT License](LICENSE).

## Ways to contribute

- 🐛 **Bugs** — open an issue with a minimal repro (request, model, expected vs actual).
- 🔌 **New providers** — add an adapter (see [Adding a provider](#adding-a-provider)).
- ✨ **Features** — please open an issue to discuss non-trivial changes first.
- 📝 **Docs** — fixes and clarifications are always welcome.

## Local setup

Requires Python 3.11+, Docker (for Postgres + Redis), and git.

```bash
git clone https://github.com/sreekarp/omnigate.git
cd omnigate

python -m venv .venv
source .venv/Scripts/activate          # Windows; use source .venv/bin/activate on macOS/Linux
pip install -e ".[dev]" -e ./sdk       # server (editable) + dev tools + the SDK

cp .env.example .env                    # set SECRET_KEY + ADMIN_API_KEY
docker compose up -d db redis           # Postgres + Redis
alembic upgrade head                    # create the schema

uvicorn app.main:app --reload           # http://localhost:8000  (/docs for the OpenAPI UI)
```

## Running the tests

The pytest suite is **fully offline** — no DB or Redis required (providers are
exercised via `httpx.MockTransport`, Postgres SQL is compile-checked):

```bash
pytest -q                               # server tests
pytest -q sdk/tests                     # SDK tests
```

There is also a **live end-to-end smoke test** that drives the whole pipeline
against a running Postgres + Redis (it mocks provider HTTP, so no real keys are
needed). Bring up `db` + `redis` first, then:

```bash
python -m scripts.smoke_e2e
```

Please make sure all three are green before opening a PR.

## Code style

OmniGate is async-first and strictly typed. Match the surrounding code:

- **Async everywhere** — `httpx.AsyncClient`, `redis.asyncio`, async SQLAlchemy sessions. Never block the event loop, and never run concurrent queries on a single async session.
- **Type hints on all functions.** Pydantic v2 for request/response schemas.
- **Logging, not `print`** — use `app.logging_config.get_logger(__name__)` inside `app/`. (`app/cli.py` is the only place `print()` is acceptable, since it's user-facing.)
- Keep additions **backward compatible** — schema fields are additive, and the `/v1/chat` + OpenAI-compatible wire formats are stable.
- The architecture and request pipeline are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Adding a provider

OmniGate providers implement a single interface, so adding one is self-contained:

1. **Implement the adapter** in `app/providers/<name>.py` as a subclass of
   `AbstractProvider` (`app/providers/base.py`):
   - `async def chat(self, request, api_key) -> ChatResponse`
   - `def stream(self, request, api_key) -> AsyncIterator[StreamChunk]` (an
     `async def` generator: yield content chunks, then **exactly one** terminal
     chunk carrying the final `Usage` + `finish_reason`).
   - Raise `ProviderError(message, status_code, retry_after=...)` on failures.
2. **Register routing** in `app/providers/registry.py`: add your model-name
   prefix to `provider_name_for_model()` and a cached factory in `get_provider()`.
3. **Add pricing** rows to `app/services/pricing.py` (USD per 1K tokens) so cost
   tracking and budgets work.
4. **Add tests** in `tests/test_providers_<name>.py` using `httpx.MockTransport`
   (assert payload shape, response parsing, and streaming usage accumulation).
5. If the provider needs non-secret config (like Azure's endpoint/deployment),
   store it in `ProviderCredential.meta` and resolve it in
   `app/services/routing.py::resolve_provider`.

## Database migrations

Models live in `app/models/db.py`. After changing them:

```bash
alembic revision -m "describe change"    # then edit the generated file
alembic upgrade head                     # apply
alembic downgrade -1 && alembic upgrade head   # verify it round-trips
```

Keep revision ids short (the `alembic_version` column is `varchar(32)`), and add
`server_default=...` for any new NOT NULL column on a populated table.

## Commits & pull requests

- Use clear, imperative commit messages; a `type(scope): summary` prefix
  (`feat`, `fix`, `docs`, `ci`, `refactor`, `test`) is appreciated.
- One logical change per PR; keep the diff focused.
- Ensure `pytest -q`, `pytest -q sdk/tests`, and (ideally) the smoke test pass.
- Update the docs / `docs/CHANGELOG.md` when behavior changes.

## Security

- **Never commit secrets.** `.env` is git-ignored; only `.env.example`
  (placeholders) is tracked.
- Provider keys are encrypted at rest (Fernet). Don't log keys or tokens.
- To report a vulnerability, please open a private security advisory on GitHub
  rather than a public issue.

Thanks for contributing! 🛰️
