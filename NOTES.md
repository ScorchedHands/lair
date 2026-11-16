# NOTES

## Worker verification policy (owner 2026-11-02)

- Per-slice: run only the targeted category tests (`--filter Category=<new>`) + build.
- Full `dotnet test` suite: once at phase close (before ship), not per slice.

## Documentation maintenance (owner 2026-11-16)

- User-facing surface: `docs/README.md` (overview + operation), `docs/reference/CLI.md` (every command, flag, output, exit code), `docs/reference/CONFIG.md` (every `lair.yaml` key, default, constraint, mutation class). Root `README.md` stays the repository entry point and links into `docs/`.
- Update the docs at every phase ship, in the same change as the ship: new commands/flags go to the CLI reference; new or changed keys, defaults, constraints, or mutation classes go to the config reference; new features go to the base page and its feature sections.
- Unshipped features are documented with an explicit status label (`in progress` / `planned` / design tentative) in the base page's Upcoming table and in the two reference pages' Planned sections. When a phase ships, move its content out of those sections into the shipped ones — never leave a shipped feature listed as upcoming.
- Docs describe the branch they sit on. Anything verified against source beats anything copied from another doc: `lair.example.yaml`, root `README.md`, and `DEVELOP.md` are part of the documented surface and must match what the loader and CLI actually accept.

## Active

- Building MVP3 on `phase/mvp3`. Slices: B1 pipeline+shell [done], B2 event schema+stats endpoint, B3 Monitor SPA wiring, B4 Chat tab. Manual Monitor + Manual Chat UI at phase close.
