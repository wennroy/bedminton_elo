import type { Metadata, Viewport } from "next";
import * as React from "react";
import "./globals.css";
import { AppShell } from "@/components/app-shell";

export const metadata: Metadata = {
  title: "卷技术小分队🏸",
  description: "羽毛球双打 ELO 记分与排行榜",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f4f5f0" },
    { media: "(prefers-color-scheme: dark)", color: "#141815" },
  ],
};

// 首屏绘制前初始化主题：读 localStorage 'badminton:theme'，
// 无手动偏好时跟随系统；同步 dark class、color-scheme 与 theme-color，避免闪白。
const themeInitScript = `(function(){try{var s=localStorage.getItem('badminton:theme');var t=(s==='dark'||s==='light')?s:(window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');var d=document.documentElement;d.classList.toggle('dark',t==='dark');d.style.colorScheme=t;if(s){var ms=document.querySelectorAll('meta[name="theme-color"]');for(var i=0;i<ms.length;i++){ms[i].removeAttribute('media');ms[i].setAttribute('content',t==='dark'?'#141815':'#f4f5f0');}}}catch(e){}})();`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="h-full antialiased" suppressHydrationWarning>
      <body className="min-h-full bg-background text-foreground">
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
