// Bundles the page into dist/: index.html plus assets/app.js and assets/app.css.
//   node build.mjs           production build (minified)
//   node build.mjs --watch   rebuild on every change
import { copyFile, mkdir, rm } from "node:fs/promises";
import { build, context } from "esbuild";

const watch = process.argv.includes("--watch");

const options = {
  entryPoints: ["src/main.tsx"],
  bundle: true,
  outdir: "dist/assets",
  entryNames: "app",
  format: "esm",
  target: "es2022",
  jsx: "automatic",
  minify: !watch,
  sourcemap: true,
  logLevel: "info",
  define: {
    "process.env.NODE_ENV": JSON.stringify(watch ? "development" : "production"),
  },
};

await rm("dist", { recursive: true, force: true });
await mkdir("dist", { recursive: true });
await copyFile("index.html", "dist/index.html");

if (watch) {
  const builder = await context(options);
  await builder.watch();
  console.log("Watching src/ for changes. Reload the page after each rebuild.");
} else {
  await build(options);
}
