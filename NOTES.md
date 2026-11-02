# NOTES

## Worker verification policy (owner 2026-11-02)

- Per-slice: run only the targeted category tests (`--filter Category=<new>`) + build.
- Full `dotnet test` suite: once at phase close (before ship), not per slice.

## Active

- Building MVP3 on `phase/mvp3`. Slices: B1 pipeline+shell [done], B2 event schema+stats endpoint, B3 Monitor SPA wiring, B4 Chat tab. Manual Monitor + Manual Chat UI at phase close.
