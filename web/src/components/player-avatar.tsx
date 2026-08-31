"use client";

import { cn } from "@/lib/utils";

export type PlayerAvatarSize = "xs" | "sm" | "md" | "lg";

interface PlayerAvatarProps {
  name: string;
  size?: PlayerAvatarSize;
  className?: string;
}

function hashString(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) + hash + str.charCodeAt(i);
  }
  return hash;
}

function nameToColors(name: string) {
  const hue = Math.abs(hashString(name)) % 360;
  return {
    background: `hsl(${hue} 80% 86%)`,
    color: `hsl(${hue} 75% 22%)`,
  };
}

const sizeClasses: Record<PlayerAvatarSize, string> = {
  xs: "size-8 text-sm",
  sm: "size-10 text-base",
  md: "size-14 text-xl",
  lg: "size-20 text-2xl",
};

export function PlayerAvatar({
  name,
  size = "md",
  className,
}: PlayerAvatarProps) {
  const initial = name.trim().slice(0, 1) || "?";
  const { background, color } = nameToColors(name);

  return (
    <div
      className={cn(
        "inline-flex items-center justify-center rounded-full font-semibold",
        sizeClasses[size],
        className
      )}
      style={{ background, color }}
      aria-hidden="true"
    >
      {initial}
    </div>
  );
}
