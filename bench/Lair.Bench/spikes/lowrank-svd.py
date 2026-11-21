"""Exact singular spectra for dequantized expert matrices exported by the low-rank SVD spike.

Reads an index JSON written by the bench harness: every entry names a raw float32 little-endian
matrix file (row-major, rows x cols). Computes each matrix's full singular value spectrum with
numpy.linalg.svd and writes one result JSON back to the harness. No derived statistics are
computed here on purpose: rank-energy fractions, thresholds, aggregates, and storage estimates
are all derived in the C# harness so every published number shares one arithmetic path.

Requires numpy; a hard import failure exits nonzero with a clear message rather than approximating.

usage: lowrank-svd.py <index.json> <spectra-out.json>
"""

import json
import sys


def main() -> int:
    try:
        import numpy as np
    except ImportError as exc:  # exact spectra are the whole point of this spike; never approximate them
        print(f"numpy is required for exact singular spectra and could not be imported: {exc}", file=sys.stderr)
        return 3

    if len(sys.argv) != 3:
        print("usage: lowrank-svd.py <index.json> <spectra-out.json>", file=sys.stderr)
        return 2

    index_path, out_path = sys.argv[1], sys.argv[2]
    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)

    results = []
    for entry in index["matrices"]:
        values = np.fromfile(entry["file"], dtype=np.float32)
        expected = int(entry["rows"]) * int(entry["cols"])
        if values.size != expected:
            print(
                f"matrix '{entry['id']}' file holds {values.size} float32 elements, "
                f"index declares rows*cols={expected}",
                file=sys.stderr,
            )
            return 4

        matrix = values.reshape((int(entry["rows"]), int(entry["cols"])))
        if not np.isfinite(matrix).all():
            print(f"matrix '{entry['id']}' contains non-finite dequantized values", file=sys.stderr)
            return 5

        spectrum = np.linalg.svd(matrix, compute_uv=False)  # numpy returns descending order
        singular = [float(v) for v in spectrum]
        if not all(np.isfinite(singular)):
            print(f"matrix '{entry['id']}' produced non-finite singular values", file=sys.stderr)
            return 6

        frobenius_squared = float(np.linalg.norm(matrix, "fro") ** 2)
        results.append(
            {
                "id": entry["id"],
                "rows": int(entry["rows"]),
                "cols": int(entry["cols"]),
                "singularValues": singular,
                "frobeniusNormSquared": frobenius_squared,
            }
        )

    payload = {"schemaVersion": 1, "matrices": results}
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    print(f"lowrank-svd.py computed spectra for {len(results)} matrices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
