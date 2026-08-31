"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Home, PlusCircle, Users, BarChart3, User } from "lucide-react";

const navItems = [
  { href: "/", label: "首页", icon: Home },
  { href: "/schedule", label: "配对", icon: Users },
  { href: "/weekly", label: "周报", icon: BarChart3 },
  { href: "/me", label: "我的", icon: User },
];

export function BottomNav() {
  const pathname = usePathname();

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-40 border-t bg-white/95 pb-[env(safe-area-inset-bottom)] backdrop-blur-xs">
      <div className="mx-auto flex h-16 max-w-md items-center justify-around px-2">
        {navItems.slice(0, 1).map((item) => (
          <NavLink key={item.href} item={item} pathname={pathname} />
        ))}

        <Link
          href="/record"
          className={cn(
            "flex h-14 w-14 -translate-y-3 flex-col items-center justify-center rounded-full bg-primary text-primary-foreground shadow-lg ring-4 ring-background transition-transform active:scale-95",
            pathname === "/record" && "bg-foreground"
          )}
          aria-label="记分"
        >
          <PlusCircle className="size-7" strokeWidth={2.5} />
          <span className="text-[10px] font-medium">记分</span>
        </Link>

        {navItems.slice(1).map((item) => (
          <NavLink key={item.href} item={item} pathname={pathname} />
        ))}
      </div>
    </nav>
  );
}

function NavLink({
  item,
  pathname,
}: {
  item: { href: string; label: string; icon: React.ElementType };
  pathname: string;
}) {
  const active = pathname === item.href;
  return (
    <Link
      href={item.href}
      className={cn(
        "flex flex-1 flex-col items-center justify-center gap-0.5 py-2 text-xs font-medium transition-colors",
        active ? "text-primary" : "text-muted-foreground hover:text-foreground"
      )}
    >
      <item.icon className="size-5" />
      <span>{item.label}</span>
    </Link>
  );
}
