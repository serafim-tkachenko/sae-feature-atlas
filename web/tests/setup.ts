import { execFileSync } from "node:child_process";

export default function setup() {
  execFileSync("uv", ["run", "python", "web/tests/generate_fixtures.py"], {
    cwd: "..",
    stdio: "inherit",
  });
}
