# LAIR docs

## Layout

| Path | Purpose |
|---|---|
| `README.md` | Product overview, quick start, API surface, monitor behavior, model acquisition summary |
| `DEVELOP.md` | Build/test/parity/bench workflow for the repository |
| `docs/reference/CLI.md` | Native `lair` executable commands and options |
| `docs/reference/CONFIG.md` | Current YAML configuration schema, validation vocabulary, hot-reload classes |

## Start here

1. Build with `dotnet build Lair.sln -c Release`.
2. Copy `lair.example.yaml` to `lair.yaml`.
3. Run the built native executable from the repository root:

```powershell
$LAIR = ".\src\Lair.Host\bin\Release\net10.0\lair.exe"
& $LAIR serve -c lair.yaml
```

On Linux/macOS, use `./src/Lair.Host/bin/Release/net10.0/lair` instead of the `.exe` path shown above.

Relative model paths resolve against the current working directory, so keep relative config/model paths rooted where you start LAIR or edit them to absolute paths before serving.
