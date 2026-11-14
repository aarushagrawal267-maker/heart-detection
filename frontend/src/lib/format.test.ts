import { describe, expect, it } from "vitest";
import { clock, duration, int, minMaxDecimate, nearestByTime, num, pct } from "./format";

describe("format helpers", () => {
  it("formats percentages and numbers, with dashes for missing values", () => {
    expect(pct(0.1234)).toBe("12.3%");
    expect(pct(null)).toBe("—");
    expect(num(0.123456, 2)).toBe("0.12");
    expect(num(undefined)).toBe("—");
    expect(int(109445)).toBe("109,445");
  });

  it("formats ECG clock time", () => {
    expect(clock(0)).toBe("0:00.0");
    expect(clock(312.25, 2)).toBe("5:12.25");
    expect(clock(1805, 0)).toBe("30:05");
  });

  it("formats durations", () => {
    expect(duration(12.34)).toBe("12.3 s");
    expect(duration(404)).toBe("6 min 44 s");
  });
});

describe("minMaxDecimate", () => {
  it("keeps short series unchanged", () => {
    const t = [0, 1, 2];
    const y = [1, 2, 3];
    expect(minMaxDecimate(t, y, 10)).toEqual([t, y]);
  });

  it("preserves extremes (QRS peaks) when downsampling", () => {
    const n = 10000;
    const t = Array.from({ length: n }, (_, i) => i / 360);
    const y = Array.from({ length: n }, () => 0);
    y[1234] = 2.5;
    y[7777] = -1.5;
    const [dt, dy] = minMaxDecimate(t, y, 100);
    expect(dy.length).toBe(200);
    expect(Math.max(...dy)).toBe(2.5);
    expect(Math.min(...dy)).toBe(-1.5);
    expect(dt[dy.indexOf(2.5)]).toBeCloseTo(1234 / 360);
  });
});

describe("nearestByTime", () => {
  const beats = [0.5, 1.3, 2.1, 2.9].map((time_s) => ({ time_s }));
  it("finds the closest beat", () => {
    expect(nearestByTime(beats, 0)).toBe(0);
    expect(nearestByTime(beats, 1.6)).toBe(1);
    expect(nearestByTime(beats, 1.8)).toBe(2);
    expect(nearestByTime(beats, 10)).toBe(3);
    expect(nearestByTime([], 1)).toBe(-1);
  });
});
