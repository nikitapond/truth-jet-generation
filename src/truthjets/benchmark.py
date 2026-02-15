from __future__ import annotations

import time


class Benchmark:
    """Optional per-stage timing for the truth-jet pipeline.

    When *enabled* is False every method is a no-op, so callers can
    unconditionally instrument the loop without branching.

    Sub-stages are supported via ``/``-delimited names, e.g.
    ``pre_clustering/SoftKillerModule``.  Starting a sub-stage pauses
    the parent timer; stopping the sub-stage resumes it.  Sub-stages
    appear indented under their parent in the summary report.
    """

    STAGES = [
        "event_generation",
        "pileup_overlay",
        "pre_clustering",
        "jet_clustering",
        "post_clustering",
        "h5_writing",
    ]

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._totals: dict[str, float] = {}
        self._batch_times: dict[str, float] = {}
        self._n_batches = 0
        self._stack: list[tuple[str, float]] = []  # (stage_name, start_time)

    def start(self, stage: str) -> None:
        """Begin timing *stage*.

        If a parent stage is already running, it is paused (its
        elapsed time so far is accumulated) and resumed when this
        sub-stage stops.
        """
        if not self.enabled:
            return
        # Pause the current stage if one is running
        if self._stack:
            parent_name, parent_t0 = self._stack[-1]
            elapsed = time.perf_counter() - parent_t0
            self._batch_times[parent_name] = (
                self._batch_times.get(parent_name, 0.0) + elapsed
            )
        self._stack.append((stage, time.perf_counter()))

    def stop(self, stage: str | None = None) -> None:
        """Stop timing the current (or named) stage.

        If a parent stage was paused, it is resumed.
        """
        if not self.enabled:
            return
        if not self._stack:
            return
        t1 = time.perf_counter()
        name, t0 = self._stack.pop()
        dt = t1 - t0
        self._batch_times[name] = self._batch_times.get(name, 0.0) + dt
        # Resume the parent stage timer
        if self._stack:
            # Reset parent start time to now
            parent_name, _ = self._stack[-1]
            self._stack[-1] = (parent_name, time.perf_counter())

    def _stage_total(self, stage: str) -> float:
        """Return the inclusive total for *stage* (overhead + sub-stages)."""
        overhead = self._totals.get(stage, 0.0)
        subs = sum(t for k, t in self._totals.items() if k.startswith(stage + "/"))
        return overhead + subs

    def end_batch(self) -> None:
        """Finalise the current batch: print a timing line and accumulate."""
        if not self.enabled:
            return
        self._n_batches += 1
        # Accumulate all entries
        for key, t in self._batch_times.items():
            self._totals[key] = self._totals.get(key, 0.0) + t
        # Build per-batch line with inclusive parent times
        parts = []
        for stage in self.STAGES:
            overhead = self._batch_times.get(stage, 0.0)
            subs = sum(t for k, t in self._batch_times.items() if k.startswith(stage + "/"))
            inclusive = overhead + subs
            if inclusive > 0:
                parts.append(f"{stage}={inclusive:.3f}s")
        total = sum(t for k, t in self._batch_times.items() if "/" not in k)
        total += sum(t for k, t in self._batch_times.items() if "/" in k)
        print(f"  [bench] batch {self._n_batches}: {' | '.join(parts)} | total={total:.3f}s")
        self._batch_times.clear()

    def report(self) -> None:
        """Print a summary table of accumulated timings."""
        if not self.enabled or not self._totals:
            return
        grand = sum(self._stage_total(s) for s in self.STAGES if s in self._totals
                     or any(k.startswith(s + "/") for k in self._totals))
        print("\n--- Benchmark Summary ---")
        print(f"{'Stage':<30s} {'Total (s)':>10s} {'Mean/batch (s)':>15s} {'%':>6s}")
        print("-" * 65)
        for stage in self.STAGES:
            inclusive = self._stage_total(stage)
            if inclusive == 0:
                continue
            mean = inclusive / self._n_batches if self._n_batches else 0.0
            pct = 100.0 * inclusive / grand if grand > 0 else 0.0
            print(f"{stage:<30s} {inclusive:>10.3f} {mean:>15.3f} {pct:>6.1f}")
            # Print sub-stages with non-negligible time, indented
            for key in sorted(self._totals):
                if key.startswith(stage + "/"):
                    st = self._totals[key]
                    if st < 0.001:
                        continue
                    sub_name = key.split("/", 1)[1]
                    sm = st / self._n_batches if self._n_batches else 0.0
                    print(f"  {sub_name:<28s} {st:>10.3f} {sm:>15.3f}")
        print("-" * 65)
        mean = grand / self._n_batches if self._n_batches else 0.0
        print(f"{'TOTAL':<30s} {grand:>10.3f} {mean:>15.3f} {'100.0':>6s}")
