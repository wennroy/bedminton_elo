"use client";

import * as React from "react";

interface ScoreInputProps {
  /** 0 = A 队，1 = B 队（决定队色与 aria 文案）。 */
  team: 0 | 1;
  /** 当前比分；NaN 表示输入框留空。 */
  score: number;
  onScoreChange: (next: number) => void;
  disabled?: boolean;
}

/**
 * 大数字比分输入 + 加/减一分按钮。
 * 从 record-form 按职责原样提取，手动记分（/record）与项目场次记分共用，
 * 视觉、取值范围（0–99）与交互保持不变。
 */
export function ScoreInput({ team, score, onScoreChange, disabled }: ScoreInputProps) {
  const label = team === 0 ? "A" : "B";

  function adjust(delta: number) {
    onScoreChange(
      Math.max(0, Math.min(99, (Number.isFinite(score) ? score : 0) + delta))
    );
  }

  return (
    <div>
      <input
        type="number"
        inputMode="numeric"
        min={0}
        max={99}
        step={1}
        disabled={disabled}
        value={Number.isFinite(score) ? score : ""}
        onChange={(e) => {
          const raw = e.target.value;
          onScoreChange(raw === "" ? NaN : Number(raw));
        }}
        aria-label={`${label}队比分`}
        className="w-full min-w-0 appearance-none border-0 bg-transparent py-3 text-center font-num text-[84px] leading-none font-bold tracking-[-4px] outline-none min-[761px]:py-[17px] min-[761px]:pb-3 min-[761px]:text-[103px] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
        style={{
          color: team === 0 ? "var(--team-a)" : "var(--team-b)",
          MozAppearance: "textfield",
        }}
      />
      <div className="flex justify-center gap-[9px] min-[761px]:gap-2.5">
        <button
          type="button"
          onClick={() => adjust(-1)}
          disabled={disabled || !(score > 0)}
          aria-label={`${label}队减一分`}
          className="h-[46px] w-[54px] rounded-lg border border-[#617057] text-[22px] text-[#f0f4e9] transition-colors hover:bg-[#ffffff0d] disabled:opacity-40 min-[761px]:h-11 min-[761px]:w-[58px]"
        >
          −
        </button>
        <button
          type="button"
          onClick={() => adjust(1)}
          disabled={disabled || !(score < 99)}
          aria-label={`${label}队加一分`}
          className="h-[46px] w-[54px] rounded-lg border border-[#617057] bg-[#ffffff12] text-[22px] text-[#f0f4e9] transition-colors hover:bg-[#ffffff1f] disabled:opacity-40 min-[761px]:h-11 min-[761px]:w-[58px]"
        >
          +
        </button>
      </div>
    </div>
  );
}
