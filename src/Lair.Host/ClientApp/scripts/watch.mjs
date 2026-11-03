import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import esbuild from "esbuild";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tailwindBin = path.join(root, "node_modules", ".bin", process.platform === "win32" ? "tailwindcss.cmd" : "tailwindcss");

const context = await esbuild.context({
  entryPoints: [path.join(root, "src", "main.tsx")],
  bundle: true,
  sourcemap: true,
  outfile: path.join(root, "dist", "app.js"),
});
await context.watch();

const tailwind = spawn(
  tailwindBin,
  ["-i", path.join(root, "src", "app.css"), "-o", path.join(root, "dist", "app.css"), "--watch"],
  { cwd: root, stdio: "inherit", shell: process.platform === "win32" },
);

function shutdown() {
  tailwind.kill();
  context.dispose().finally(() => process.exit(0));
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
tailwind.on("exit", (code) => {
  context.dispose().finally(() => process.exit(code ?? 0));
});
