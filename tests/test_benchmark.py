from __future__ import annotations

import time

from truthjets.benchmark import Benchmark


class TestBenchmarkDisabled:
    def test_disabled_is_noop(self, capsys):
        """Disabled benchmark should produce no output and track no state."""
        bench = Benchmark(enabled=False)
        bench.start("event_generation")
        bench.stop("event_generation")
        bench.end_batch()
        bench.report()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert bench._totals == {}
        assert bench._n_batches == 0


class TestBenchmarkEnabled:
    def test_timing_accumulates_across_batches(self):
        """Totals should accumulate across multiple batches."""
        bench = Benchmark(enabled=True)

        for _ in range(3):
            bench.start("event_generation")
            time.sleep(0.01)
            bench.stop("event_generation")
            bench.end_batch()

        assert bench._n_batches == 3
        assert "event_generation" in bench._totals
        assert bench._totals["event_generation"] > 0.02

    def test_skipped_stages_omitted(self, capsys):
        """Stages that never fire should not appear in output."""
        bench = Benchmark(enabled=True)
        bench.start("jet_clustering")
        bench.stop("jet_clustering")
        bench.end_batch()
        bench.report()

        captured = capsys.readouterr()
        assert "jet_clustering" in captured.out
        assert "event_generation" not in captured.out
        assert "pileup_overlay" not in captured.out

    def test_report_prints_summary_table(self, capsys):
        """Report should print header, stage lines, and total."""
        bench = Benchmark(enabled=True)
        bench.start("event_generation")
        bench.stop("event_generation")
        bench.start("jet_clustering")
        bench.stop("jet_clustering")
        bench.end_batch()
        bench.report()

        captured = capsys.readouterr()
        assert "Benchmark Summary" in captured.out
        assert "event_generation" in captured.out
        assert "jet_clustering" in captured.out
        assert "TOTAL" in captured.out

    def test_end_batch_prints_per_batch_line(self, capsys):
        """end_batch should print a per-batch timing line."""
        bench = Benchmark(enabled=True)
        bench.start("h5_writing")
        bench.stop("h5_writing")
        bench.end_batch()

        captured = capsys.readouterr()
        assert "[bench] batch 1:" in captured.out
        assert "h5_writing=" in captured.out

    def test_stop_without_stage_uses_current(self):
        """stop() with no argument should use the stage from start()."""
        bench = Benchmark(enabled=True)
        bench.start("pre_clustering")
        bench.stop()
        bench.end_batch()
        assert "pre_clustering" in bench._totals

    def test_multiple_stages_per_batch(self, capsys):
        """Multiple stages in one batch should all appear in output."""
        bench = Benchmark(enabled=True)
        bench.start("event_generation")
        bench.stop()
        bench.start("pileup_overlay")
        bench.stop()
        bench.start("jet_clustering")
        bench.stop()
        bench.end_batch()

        captured = capsys.readouterr()
        assert "event_generation=" in captured.out
        assert "pileup_overlay=" in captured.out
        assert "jet_clustering=" in captured.out


class TestBenchmarkSubStages:
    def test_sub_stages_appear_in_report(self, capsys):
        """Sub-stages should appear indented under their parent."""
        bench = Benchmark(enabled=True)
        bench.start("post_clustering")
        bench.start("post_clustering/LabelModule")
        time.sleep(0.01)
        bench.stop("post_clustering/LabelModule")
        bench.start("post_clustering/FilterModule")
        time.sleep(0.01)
        bench.stop("post_clustering/FilterModule")
        bench.stop("post_clustering")
        bench.end_batch()
        bench.report()

        captured = capsys.readouterr()
        assert "post_clustering" in captured.out
        assert "LabelModule" in captured.out
        assert "FilterModule" in captured.out

    def test_sub_stages_dont_double_count_parent(self):
        """Parent time should exclude sub-stage time (no double counting)."""
        bench = Benchmark(enabled=True)
        bench.start("pre_clustering")
        time.sleep(0.01)
        bench.start("pre_clustering/SlowModule")
        time.sleep(0.05)
        bench.stop("pre_clustering/SlowModule")
        time.sleep(0.01)
        bench.stop("pre_clustering")
        bench.end_batch()

        # Parent time should be ~0.02s (the time outside sub-stages)
        # Sub-stage time should be ~0.05s
        parent = bench._totals["pre_clustering"]
        sub = bench._totals["pre_clustering/SlowModule"]
        assert sub > parent, "sub-stage should be longer than parent overhead"
        # Parent + sub should roughly equal total wall time (~0.07s)
        assert parent + sub > 0.05

    def test_sub_stages_excluded_from_grand_total(self):
        """Sub-stage times should not inflate the grand total."""
        bench = Benchmark(enabled=True)
        bench.start("jet_clustering")
        time.sleep(0.01)
        bench.stop()
        bench.start("post_clustering")
        bench.start("post_clustering/Mod")
        time.sleep(0.01)
        bench.stop("post_clustering/Mod")
        bench.stop("post_clustering")
        bench.end_batch()

        grand = sum(t for k, t in bench._totals.items() if "/" not in k)
        total_with_subs = sum(bench._totals.values())
        assert total_with_subs > grand, "sub-stages add to total dict but not grand"

    def test_sub_stages_not_in_batch_line(self, capsys):
        """Per-batch line should only show parent stages, not sub-stages."""
        bench = Benchmark(enabled=True)
        bench.start("pre_clustering")
        bench.start("pre_clustering/SoftKillerModule")
        bench.stop("pre_clustering/SoftKillerModule")
        bench.stop("pre_clustering")
        bench.end_batch()

        captured = capsys.readouterr()
        assert "pre_clustering=" in captured.out
        assert "SoftKillerModule" not in captured.out
