import { copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

await mkdir(path.join(root, "dist"), { recursive: true });
await copyFile(path.join(root, "index.html"), path.join(root, "dist", "index.html"));
