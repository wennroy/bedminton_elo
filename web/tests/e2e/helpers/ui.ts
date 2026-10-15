import type { Locator, Page } from "playwright";
import { expect } from "vitest";

/**
 * 不用 @playwright/test 的 expect 扩展，而用 waitFor + vitest expect
 * 封装常用 UI 断言（仓库只加 playwright 库本体，保持依赖最小）。
 * waitFor 自带重试，语义与 Playwright 断言一致。
 */

export async function visible(locator: Locator): Promise<void> {
  await locator.waitFor({ state: "visible" });
  expect(await locator.isVisible()).toBe(true);
}

export async function hidden(locator: Locator): Promise<void> {
  await locator.waitFor({ state: "hidden" });
}

/** 元素文本（规整空白后）完全等于 expected。 */
export async function textIs(locator: Locator, expected: string): Promise<void> {
  await locator.waitFor({ state: "visible" });
  const content = (await locator.textContent()) ?? "";
  expect(content.replace(/\s+/g, " ").trim()).toBe(expected);
}

/** 元素文本包含子串或匹配正则。 */
export async function textMatches(locator: Locator, pattern: string | RegExp): Promise<void> {
  await locator.waitFor({ state: "visible" });
  const content = (await locator.textContent()) ?? "";
  if (typeof pattern === "string") {
    expect(content).toContain(pattern);
  } else {
    expect(content).toMatch(pattern);
  }
}

/** 页面 URL 匹配正则。 */
export async function urlIs(page: Page, pattern: RegExp): Promise<void> {
  await page.waitForURL(pattern);
  expect(page.url()).toMatch(pattern);
}

export async function disabled(locator: Locator): Promise<void> {
  await locator.waitFor({ state: "attached" });
  expect(await locator.isDisabled()).toBe(true);
}

export async function attrIs(
  locator: Locator,
  name: string,
  value: string
): Promise<void> {
  await locator.waitFor({ state: "visible" });
  expect(await locator.getAttribute(name)).toBe(value);
}

export async function inputValueIs(locator: Locator, value: string): Promise<void> {
  await locator.waitFor({ state: "visible" });
  expect(await locator.inputValue()).toBe(value);
}

/** 定位器匹配数 === 0（先等一小段让页面稳定）。 */
export async function countIsZero(locator: Locator): Promise<void> {
  await expect
    .poll(async () => locator.count(), { timeout: 10000, interval: 200 })
    .toBe(0);
}

/**
 * 表单错误提示：Next.js 有一个常驻的 __next-route-announcer__
 * （ visually-hidden 的 div[role=alert]），getByRole("alert") 会命中它；
 * 产品里的错误提示都是 <p role="alert">，据此区分。
 */
export function formAlert(page: Page): Locator {
  return page.locator("p[role='alert']");
}
