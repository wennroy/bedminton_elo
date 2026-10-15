import { chromium, type Browser, type BrowserContext } from "playwright";

/**
 * 真实 Chromium 旅程自动化。
 * 身份通过 localStorage `badminton:myPlayerId` 注入（与 AppShell/identity.ts
 * 同一机制），A/B 两个身份各用独立 BrowserContext（隔离存储）。
 */

export interface BrowserHandle {
  browser: Browser;
  /** 以指定球员身份打开独立上下文（playerId 为 null 表示未选身份）。 */
  newIdentityContext(playerId: number | null): Promise<BrowserContext>;
  close(): Promise<void>;
}

export async function launchBrowser(): Promise<BrowserHandle> {
  const browser = await chromium.launch();
  return {
    browser,
    async newIdentityContext(playerId: number | null): Promise<BrowserContext> {
      const context = await browser.newContext();
      if (playerId !== null) {
        await context.addInitScript((id: number) => {
          window.localStorage.setItem("badminton:myPlayerId", String(id));
        }, playerId);
      }
      return context;
    },
    close: () => browser.close(),
  };
}
