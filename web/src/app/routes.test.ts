import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

// Every server-component page reads the sqlite db, which only exists at
// runtime. Without `export const dynamic = "force-dynamic"` Next.js
// prerenders the page at build time against an empty db and serves stale
// static HTML in production (this bit us: /record shipped with zero players).
function collectPages(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      out.push(...collectPages(full));
    } else if (entry === "page.tsx") {
      out.push(full);
    }
  }
  return out;
}

describe("route rendering mode", () => {
  const appDir = join(__dirname, "..");
  const pages = collectPages(appDir);

  it("finds pages to check", () => {
    expect(pages.length).toBeGreaterThan(0);
  });

  for (const page of pages) {
    const rel = page.replace(`${appDir}/`, "app/");
    const source = readFileSync(page, "utf8");
    const isClientComponent = /^\s*["']use client["']/.test(source);

    if (isClientComponent) {
      it(`${rel} is a client component (fetches at runtime)`, () => {
        expect(isClientComponent).toBe(true);
      });
    } else {
      it(`${rel} server component opts out of static prerendering`, () => {
        expect(source).toContain('export const dynamic = "force-dynamic"');
      });
    }
  }
});
