"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { DropdownMenu } from "radix-ui";
import {
  LayoutGrid,
  Users,
  TrendingUp,
  CalendarCheck,
  Plus,
  Sun,
  Moon,
  Shuffle,
  Newspaper,
  Target,
  UserRound,
  ScrollText,
  Shield,
  ChevronsUpDown,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getMyPlayerId } from "@/lib/identity";
import { IdentityPicker } from "@/components/identity-picker";
import { PlayerAvatar } from "@/components/player-avatar";

const THEME_KEY = "badminton:theme";
const LIGHT_BG = "#f4f5f0";
const DARK_BG = "#141815";

type Theme = "light" | "dark";

interface PlayerInfo {
  id: number;
  name: string;
}

const primaryNav = [
  { href: "/", label: "俱乐部总览", icon: LayoutGrid },
  { href: "/players", label: "球员分析", icon: Users },
  { href: "/trends", label: "全员趋势", icon: TrendingUp },
  { href: "/signup", label: "每周报名", icon: CalendarCheck },
  { href: "/record", label: "记一场比赛", icon: Plus },
];

const secondaryNav = [
  { href: "/schedule", label: "配对", icon: Shuffle },
  { href: "/weekly", label: "周报", icon: Newspaper },
  { href: "/predict", label: "预测", icon: Target },
  { href: "/me", label: "我的", icon: UserRound },
  { href: "/changelog", label: "更新日志", icon: ScrollText },
  { href: "/admin", label: "管理", icon: Shield },
];

const pageTitles: [RegExp, string][] = [
  [/^\/$/, "俱乐部总览"],
  [/^\/players/, "球员分析"],
  [/^\/trends/, "全员趋势"],
  [/^\/signup/, "每周报名"],
  [/^\/record/, "记一场比赛"],
  [/^\/schedule/, "配对"],
  [/^\/weekly/, "周报"],
  [/^\/predict/, "预测"],
  [/^\/me/, "我的"],
  [/^\/changelog/, "更新日志"],
  [/^\/admin/, "管理"],
];

function isActive(pathname: string, href: string) {
  return href === "/" ? pathname === "/" : pathname.startsWith(href);
}

function paintTheme(next: Theme, pinMeta: boolean) {
  const root = document.documentElement;
  root.classList.toggle("dark", next === "dark");
  root.style.colorScheme = next;
  if (pinMeta) {
    document
      .querySelectorAll('meta[name="theme-color"]')
      .forEach((meta) => {
        meta.removeAttribute("media");
        meta.setAttribute("content", next === "dark" ? DARK_BG : LIGHT_BG);
      });
  }
}

function BrandMark({ className, iconClassName }: { className?: string; iconClassName?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "grid shrink-0 rotate-[-7deg] place-items-center rounded-[11px] bg-primary text-primary-foreground",
        className
      )}
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.65}
        strokeLinecap="round"
        strokeLinejoin="round"
        className={cn("size-[25px] rotate-[7deg]", iconClassName)}
      >
        <rect x="3" y="4" width="18" height="16" rx="1" />
        <path d="M12 4v16M3 12h18M6 4v16M18 4v16" />
      </svg>
    </span>
  );
}

function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <Link href="/" className="flex items-center gap-2.5" aria-label="卷技术小分队首页">
      <BrandMark
        className={compact ? "size-[31px] rounded-[9px]" : "size-[38px]"}
        iconClassName={compact ? "size-[22px]" : undefined}
      />
      <span>
        <span className={cn("block font-bold tracking-[-0.3px] leading-[1.4]", compact ? "text-[13px]" : "text-[15px]")}>
          卷技术小分队
        </span>
        <span className={cn("block tracking-[3px] text-muted-foreground", compact ? "text-[8px]" : "text-[9px]")}>
          COURTSIDE CLUB
        </span>
      </span>
    </Link>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [theme, setTheme] = React.useState<Theme>("light");
  const [players, setPlayers] = React.useState<PlayerInfo[]>([]);
  const [myId, setMyId] = React.useState<number | null>(null);
  const [pickerMounted, setPickerMounted] = React.useState(false);
  const pickerTriggerRef = React.useRef<HTMLButtonElement>(null);
  const clickTriggerAfterMount = React.useRef(false);

  // 初始化主题状态；未手动选择时响应系统主题变化
  React.useEffect(() => {
    setTheme(document.documentElement.classList.contains("dark") ? "dark" : "light");
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (event: MediaQueryListEvent) => {
      let stored: string | null = null;
      try {
        stored = localStorage.getItem(THEME_KEY);
      } catch {
        stored = null;
      }
      if (!stored) {
        const next: Theme = event.matches ? "dark" : "light";
        paintTheme(next, false);
        setTheme(next);
      }
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  // 读取当前身份并拉取球员列表；路由变化时同步（页面内可能换过身份）
  React.useEffect(() => {
    setMyId(getMyPlayerId());
    let cancelled = false;
    fetch("/api/players")
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => {
        if (!cancelled && Array.isArray(data)) setPlayers(data);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [pathname]);

  const toggleTheme = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // 存储不可用时仅本次会话生效
    }
    paintTheme(next, true);
    setTheme(next);
  };

  const me = players.find((p) => p.id === myId) ?? null;

  const openIdentityPicker = () => {
    if (pickerMounted) {
      pickerTriggerRef.current?.click();
      return;
    }
    // IdentityPicker 在无身份时会自动打开；已有身份则挂载后补一次触发点击
    clickTriggerAfterMount.current = getMyPlayerId() !== null;
    setPickerMounted(true);
  };

  React.useEffect(() => {
    if (pickerMounted && clickTriggerAfterMount.current) {
      clickTriggerAfterMount.current = false;
      pickerTriggerRef.current?.click();
    }
  }, [pickerMounted]);

  const handleIdentitySelect = (id: number) => {
    setMyId(id);
    // 可能刚通过弹层新增球员，刷新名单
    fetch("/api/players")
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => {
        if (Array.isArray(data)) setPlayers(data);
      })
      .catch(() => {});
  };

  const playerMatch = pathname.match(/^\/players\/(\d+)/);
  const currentTitle =
    pageTitles.find(([pattern]) => pattern.test(pathname))?.[1] ?? "俱乐部总览";
  const currentPlayer = playerMatch
    ? players.find((p) => p.id === Number(playerMatch[1]))
    : undefined;

  const menuItemClass =
    "flex cursor-pointer items-center gap-2.5 rounded-lg px-2 py-2 text-sm text-foreground outline-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground";

  const avatarButton = (
    <>
      {me ? (
        <PlayerAvatar name={me.name} size="xs" />
      ) : (
        <span className="grid size-8 place-items-center rounded-full bg-muted text-muted-foreground">
          <UserRound className="size-4" />
        </span>
      )}
    </>
  );

  const identityMenu = (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          type="button"
          className="rounded-full outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={me ? `当前身份：${me.name}，打开账户菜单` : "尚未选择身份，打开账户菜单"}
        >
          {avatarButton}
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={8}
          className="z-50 w-56 rounded-xl border border-border bg-popover p-1.5 text-popover-foreground shadow-[0_20px_70px_rgb(0_0_0/0.18)]"
        >
          <div className="flex items-center gap-2.5 px-2 py-2">
            {avatarButton}
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold">
                {me ? me.name : "未选择身份"}
              </div>
              <div className="text-[11px] text-muted-foreground">当前身份</div>
            </div>
          </div>
          <DropdownMenu.Item className={menuItemClass} onSelect={openIdentityPicker}>
            <ChevronsUpDown className="size-4 text-muted-foreground" />
            切换身份
          </DropdownMenu.Item>
          <DropdownMenu.Separator className="my-1 h-px bg-border" />
          {secondaryNav.map((item) => (
            <DropdownMenu.Item key={item.href} asChild className={menuItemClass}>
              <Link href={item.href}>
                <item.icon className="size-4 text-muted-foreground" />
                {item.label}
              </Link>
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );

  const themeToggle = (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={theme === "dark" ? "切换浅色模式" : "切换深色模式"}
      title={theme === "dark" ? "切换浅色模式" : "切换深色模式"}
      className="grid size-[35px] place-items-center rounded-full border border-border bg-card transition-colors hover:bg-secondary min-[761px]:size-10"
    >
      {theme === "dark" ? (
        <Sun className="size-[18px]" strokeWidth={1.65} />
      ) : (
        <Moon className="size-[18px]" strokeWidth={1.65} />
      )}
    </button>
  );

  const sideNavItem = (item: (typeof primaryNav)[number]) => {
    const active = isActive(pathname, item.href);
    return (
      <Link
        key={item.href}
        href={item.href}
        title={item.label}
        aria-current={active ? "page" : undefined}
        className={cn(
          "flex h-12 w-12 items-center justify-center rounded-[10px] font-[550] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
          "min-[1191px]:h-auto min-[1191px]:w-auto min-[1191px]:justify-start min-[1191px]:gap-[13px] min-[1191px]:px-[14px] min-[1191px]:py-3",
          active && "bg-primary text-primary-foreground hover:bg-primary hover:text-primary-foreground"
        )}
      >
        <item.icon className="size-5 shrink-0" strokeWidth={1.65} />
        <span className="hidden min-[1191px]:inline">{item.label}</span>
        {item.href === "/players" && players.length > 0 && (
          <span className="ml-auto hidden text-[11px] opacity-75 min-[1191px]:inline">
            {String(players.length).padStart(2, "0")}
          </span>
        )}
      </Link>
    );
  };

  const bottomItem = (
    href: string,
    label: string,
    Icon: typeof LayoutGrid,
    active: boolean
  ) => (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={cn(
        "flex min-h-12 flex-col items-center justify-center gap-[3px] text-[10px]",
        active ? "font-bold text-win" : "text-muted-foreground"
      )}
    >
      <Icon className="size-[21px]" strokeWidth={1.65} />
      <span>{label}</span>
    </Link>
  );

  return (
    <>
      {/* 桌面 / 平板侧栏：≥1191px 文字侧栏，761–1190px 图标侧栏，手机隐藏 */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[76px] flex-col items-center border-r border-border bg-card px-[13px] py-[25px] min-[761px]:flex min-[1191px]:w-[222px] min-[1191px]:items-stretch min-[1191px]:px-[22px] min-[1191px]:pt-8 min-[1191px]:pb-6">
        <div className="hidden min-[1191px]:block">
          <Brand />
        </div>
        <div className="min-[1191px]:hidden">
          <BrandMark className="size-[38px]" />
        </div>
        <div className="mt-[47px] hidden px-3 pb-3 text-[10px] tracking-[1.5px] text-muted-foreground min-[1191px]:block">
          YOUR BADMINTON CLUB
        </div>
        <nav aria-label="主导航" className="mt-[50px] grid gap-3 min-[1191px]:mt-0 min-[1191px]:gap-[7px]">
          {primaryNav.map(sideNavItem)}
          <div className="mx-auto my-1 h-px w-6 bg-border min-[1191px]:mx-3 min-[1191px]:w-auto" />
          {secondaryNav.map(sideNavItem)}
        </nav>
        <div className="mt-auto hidden min-[1191px]:block">
          <div className="px-3 pt-5 pb-5">
            <svg
              viewBox="0 0 84 44"
              fill="none"
              stroke="currentColor"
              aria-hidden="true"
              className="mb-4 h-11 w-[84px] text-muted-foreground opacity-45"
            >
              <rect x="0.5" y="0.5" width="83" height="43" />
              <rect x="6.5" y="6.5" width="71" height="31" />
              <line x1="42" y1="0" x2="42" y2="44" />
            </svg>
            <p className="text-[11px] leading-[1.9] text-muted-foreground">
              卷技术小分队
              <br />
              羽毛球双打数据
            </p>
          </div>
          <button
            type="button"
            onClick={openIdentityPicker}
            aria-label={me ? `切换当前身份，${me.name}` : "选择当前身份"}
            className="flex w-full items-center gap-2.5 border-t border-border pt-5 text-left"
          >
            {avatarButton}
            <span className="min-w-0">
              <span className="block truncate text-sm font-semibold">
                {me ? me.name : "选择身份"}
              </span>
              <span className="block text-[10px] text-muted-foreground">当前身份</span>
            </span>
            <ChevronsUpDown className="ml-auto size-4 text-muted-foreground" />
          </button>
        </div>
        <button
          type="button"
          onClick={openIdentityPicker}
          aria-label={me ? `切换当前身份，${me.name}` : "选择当前身份"}
          className="mt-auto min-[1191px]:hidden"
        >
          {avatarButton}
        </button>
      </aside>

      <div className="min-[761px]:ml-[76px] min-[1191px]:ml-[222px]">
        {/* 顶栏：手机显示品牌，桌面显示面包屑；右侧为主题切换与头像菜单 */}
        <header className="flex h-[70px] items-center justify-between border-b border-border bg-background px-5 min-[761px]:h-[78px] min-[761px]:px-7 min-[1191px]:px-10">
          <div className="min-[761px]:hidden">
            <Brand compact />
          </div>
          <div className="hidden items-center gap-3.5 text-xs text-muted-foreground min-[761px]:flex">
            俱乐部 <span>/</span>{" "}
            <strong className="font-[550] text-foreground">{currentTitle}</strong>
            {currentPlayer && (
              <>
                <span>/</span> <span>{currentPlayer.name}</span>
              </>
            )}
          </div>
          <div className="flex items-center gap-[9px] min-[761px]:gap-3.5">
            {themeToggle}
            {identityMenu}
          </div>
        </header>

        <main className="mx-auto min-h-[calc(100dvh-70px)] max-w-[1480px] px-5 pt-[25px] pb-[calc(100px+env(safe-area-inset-bottom))] min-[761px]:min-h-[calc(100dvh-78px)] min-[761px]:p-7 min-[1191px]:px-10 min-[1191px]:pt-[34px] min-[1191px]:pb-8">
          {children}
        </main>
      </div>

      {/* 手机底栏：总览 / 球员 / 记一场 / 报名 */}
      <nav
        aria-label="手机主导航"
        className="fixed inset-x-0 bottom-0 z-30 grid grid-cols-4 border-t border-border bg-card px-3 pt-2 pb-[calc(8px+env(safe-area-inset-bottom))] min-[761px]:hidden"
      >
        {bottomItem("/", "总览", LayoutGrid, pathname === "/")}
        {bottomItem(
          "/players",
          "球员",
          Users,
          pathname.startsWith("/players") || pathname.startsWith("/trends")
        )}
        <Link
          href="/record"
          aria-current={pathname.startsWith("/record") ? "page" : undefined}
          className={cn(
            "flex min-h-12 flex-col items-center justify-center gap-[3px] text-[10px]",
            pathname.startsWith("/record") ? "font-bold text-win" : "text-muted-foreground"
          )}
        >
          <span className="grid h-7 w-[45px] place-items-center rounded-[9px] bg-primary text-primary-foreground">
            <Plus className="size-[21px]" strokeWidth={1.65} />
          </span>
          <span>记一场</span>
        </Link>
        {bottomItem("/signup", "报名", CalendarCheck, pathname.startsWith("/signup"))}
      </nav>

      {pickerMounted && (
        <IdentityPicker
          players={players}
          onSelect={handleIdentitySelect}
          trigger={
            <button ref={pickerTriggerRef} type="button" className="hidden" tabIndex={-1} aria-hidden="true" />
          }
        />
      )}
    </>
  );
}
