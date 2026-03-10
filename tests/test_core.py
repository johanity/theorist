"""Core tests for Theorist."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import theorist
from theorist.brain import Brain
from theorist.engine import Engine
from theorist.surprise import SurpriseNormalizer
from theorist.types import Experiment, Prediction, Results


class TestSurpriseNormalizer(unittest.TestCase):
    def test_first_observation_max_surprise(self):
        n = SurpriseNormalizer()
        n.update(5.0)
        self.assertEqual(n.surprise(0.0, 5.0), 1.0)

    def test_small_error_low_surprise(self):
        n = SurpriseNormalizer()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            n.update(v)
        self.assertLess(n.surprise(3.0, 3.1), 0.1)

    def test_large_error_high_surprise(self):
        n = SurpriseNormalizer()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            n.update(v)
        s_small = n.surprise(3.0, 3.1)
        s_large = n.surprise(3.0, 10.0)
        self.assertGreater(s_large, s_small)

    def test_welford_std(self):
        n = SurpriseNormalizer()
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            n.update(v)
        self.assertAlmostEqual(n.std, 2.14, places=1)

    def test_count(self):
        n = SurpriseNormalizer()
        self.assertEqual(n.count, 0)
        n.update(1.0)
        n.update(2.0)
        self.assertEqual(n.count, 2)


class TestBrain(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.brain = Brain(path=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_init_empty(self):
        self.assertEqual(self.brain.theory["total_experiments"], 0)
        self.assertEqual(self.brain.theory["domains_seen"], [])

    def test_start_task(self):
        self.brain.start_task("physics")
        self.assertIn("physics", self.brain.theory["domains_seen"])
        self.assertEqual(self.brain.theory["total_tasks"], 1)

    def test_start_task_idempotent(self):
        self.brain.start_task("physics")
        self.brain.start_task("physics")
        self.assertEqual(self.brain.theory["total_tasks"], 1)

    def test_log_experiment(self):
        self.brain.log_experiment({
            "domain": "math", "exp_id": 0,
            "config": {"x": 1}, "actual": 5.0,
            "surprise": 0.5, "status": "best",
        })
        self.assertEqual(self.brain.theory["total_experiments"], 1)
        self.assertTrue(self.brain.experiments_file.exists())

    def test_persistence(self):
        self.brain.start_task("physics")
        self.brain.log_experiment({"domain": "physics", "actual": 1.0, "surprise": 0.3})
        self.brain.save()

        brain2 = Brain(path=self.tmpdir)
        self.assertEqual(brain2.theory["total_experiments"], 1)
        self.assertIn("physics", brain2.theory["domains_seen"])

    def test_mid_preference_default(self):
        self.assertEqual(self.brain.mid_preference, 0.5)

    def test_mid_preference_tracked(self):
        for _ in range(10):
            self.brain.record_mid_vs_extreme(True)
        for _ in range(5):
            self.brain.record_mid_vs_extreme(False)
        self.assertAlmostEqual(self.brain.mid_preference, 10 / 15)

    def test_export_anonymized(self):
        self.brain.start_task("math")
        data = self.brain.export_anonymized()
        self.assertIn("position_scores", data)
        self.assertIn("total_experiments", data)
        self.assertIn("mid_vs_extreme", data)

    def test_apply_update_confirm(self):
        self.brain.apply_update("CONFIRM", "moderate values beat extremes")
        self.assertIn("moderate values beat extremes", self.brain.theory["confirmed"])

    def test_apply_update_meta(self):
        self.brain.apply_update("META", "diminishing returns")
        self.assertIn("diminishing returns", self.brain.theory["meta_patterns"])

    def test_apply_update_cross_domain(self):
        self.brain.apply_update("CROSS_DOMAIN", "universal pattern")
        self.assertIn("universal pattern", self.brain.theory["cross_domain"])

    def test_apply_update_refute(self):
        self.brain.apply_update("CONFIRM", "high values are always best")
        self.brain.apply_update("REFUTE", "high values")
        self.assertNotIn("high values are always best", self.brain.theory["confirmed"])

    def test_apply_update_no_duplicates(self):
        self.brain.apply_update("CONFIRM", "same thing")
        self.brain.apply_update("CONFIRM", "same thing")
        self.assertEqual(self.brain.theory["confirmed"].count("same thing"), 1)

    def test_summary(self):
        self.brain.start_task("physics")
        self.brain.apply_update("CONFIRM", "test belief")
        summary = self.brain.summary()
        self.assertIn("Theorist Brain", summary)
        self.assertIn("test belief", summary)

    def test_reset(self):
        self.brain.start_task("physics")
        self.brain.log_experiment({"domain": "physics", "actual": 1.0, "surprise": 0.3})
        self.brain.reset()
        self.assertEqual(self.brain.theory["total_experiments"], 0)

    def test_position_scores(self):
        self.brain.update_position_scores("mid", 5.0)
        self.brain.update_position_scores("mid", 3.0)
        scores = self.brain.theory["param_position_scores"]["mid"]
        self.assertEqual(scores["count"], 2)
        self.assertAlmostEqual(scores["avg_metric"], 4.0)


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.brain = Brain(path=self.tmpdir)
        self.engine = Engine(self.brain)
        self.space = {"x": [-5, -3, -1, 0, 1, 3, 5], "y": [-5, -3, -1, 0, 1, 3, 5]}
        self.baseline = {"x": 0, "y": 0}

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_first_prediction_is_baseline(self):
        self.engine.start_task(self.space, self.baseline, "math")
        pred = self.engine.predict(0)
        self.assertEqual(pred.config, self.baseline)
        self.assertEqual(pred.confidence, "low")

    def test_record_tracks_best(self):
        self.engine.start_task(self.space, self.baseline, "math")
        _, _, status = self.engine.record({"x": 0, "y": 0}, 0.0, 5.0, minimize=True)
        self.assertEqual(status, "best")
        self.assertEqual(self.engine.best_metric, 5.0)

        _, _, status = self.engine.record({"x": 1, "y": -1}, 5.0, 1.0, minimize=True)
        self.assertEqual(status, "best")
        self.assertEqual(self.engine.best_metric, 1.0)

    def test_record_discards_worse(self):
        self.engine.start_task(self.space, self.baseline, "math")
        self.engine.record({"x": 0, "y": 0}, 0.0, 5.0, minimize=True)
        _, _, status = self.engine.record({"x": -5, "y": 5}, 5.0, 85.0, minimize=True)
        self.assertEqual(status, "discard")
        self.assertEqual(self.engine.best_metric, 5.0)

    def test_record_maximize(self):
        self.engine.start_task(self.space, self.baseline, "math")
        self.engine.record({"x": 0, "y": 0}, 0.0, 5.0, minimize=False)
        _, _, status = self.engine.record({"x": 1, "y": 1}, 5.0, 10.0, minimize=False)
        self.assertEqual(status, "best")
        self.assertEqual(self.engine.best_metric, 10.0)

    def test_classify_value(self):
        self.engine.start_task(self.space, self.baseline, "math")
        self.assertEqual(self.engine.classify_value("x", -5), "low")
        self.assertEqual(self.engine.classify_value("x", 0), "mid")
        self.assertEqual(self.engine.classify_value("x", 5), "high")

    def test_predict_returns_prediction(self):
        self.engine.start_task(self.space, self.baseline, "math")
        for i in range(5):
            pred = self.engine.predict(i)
            self.assertIsInstance(pred, Prediction)
            self.assertIn("x", pred.config)
            self.assertIn("y", pred.config)
            # Record something so engine has history
            self.engine.record(pred.config, pred.predicted_metric,
                               sum(v**2 for v in pred.config.values()), minimize=True)


class TestExperimentDecorator(unittest.TestCase):
    def test_basic_optimize(self):
        tmpdir = tempfile.mkdtemp()
        try:
            @theorist.experiment(
                search_space={"x": [0, 1, 2]},
                metric="score",
                domain="test",
            )
            def simple(config):
                return {"score": (config["x"] - 1) ** 2}

            results = simple.optimize(n=5, verbose=False, brain_path=tmpdir)
            self.assertIsInstance(results, Results)
            self.assertEqual(len(results.experiments), 5)
            self.assertLessEqual(results.best_metric, 1.0)
        finally:
            shutil.rmtree(tmpdir)

    def test_decorator_preserves_function(self):
        @theorist.experiment(
            search_space={"x": [0, 1]},
            metric="v",
        )
        def my_fn(config):
            return {"v": config["x"]}

        self.assertEqual(my_fn.__name__, "my_fn")
        self.assertTrue(hasattr(my_fn, "optimize"))
        self.assertTrue(hasattr(my_fn, "_search_space"))

    def test_minimize_false(self):
        tmpdir = tempfile.mkdtemp()
        try:
            @theorist.experiment(
                search_space={"x": [0, 1, 2, 3]},
                metric="score",
                minimize=False,
                domain="maximize_test",
            )
            def maximize_fn(config):
                return {"score": config["x"] * 10}

            results = maximize_fn.optimize(n=8, verbose=False, brain_path=tmpdir)
            self.assertEqual(results.best_metric, 30.0)
        finally:
            shutil.rmtree(tmpdir)


class TestResults(unittest.TestCase):
    def test_report_format(self):
        results = Results(
            best_config={"x": 1},
            best_metric=1.0,
            experiments=[
                Experiment(0, {"x": 0}, 0.0, 5.0, 1.0, "baseline", "Baseline: 5.0", "best"),
                Experiment(1, {"x": 1}, 5.0, 1.0, 0.5, "test x=1", "improved", "best"),
                Experiment(2, {"x": 2}, 1.0, 1.0, 0.0, "test x=2", "same", "discard"),
                Experiment(3, {"x": 0}, 1.0, 5.0, 0.8, "retest", "worse", "discard"),
            ],
            theory_updates=["improved"],
            domain="test",
        )
        report = results.report()
        self.assertIn("THEORIST", report)
        self.assertIn("1.0000", report)
        self.assertIn("Surprise trajectory", report)
        self.assertIn("Learnings", report)

    def test_str(self):
        r = Results(best_config={}, best_metric=0.0, domain="x")
        self.assertIn("THEORIST", str(r))


class TestCrossDomainTransfer(unittest.TestCase):
    def test_brain_accumulates_across_tasks(self):
        tmpdir = tempfile.mkdtemp()
        try:
            @theorist.experiment(
                search_space={"a": [0, 1, 2, 3, 4]},
                metric="v",
                domain="domain_a",
            )
            def task1(config):
                return {"v": (config["a"] - 1) ** 2}

            r1 = task1.optimize(n=8, verbose=False, brain_path=tmpdir)

            @theorist.experiment(
                search_space={"b": [0, 1, 2, 3, 4]},
                metric="v",
                domain="domain_b",
            )
            def task2(config):
                return {"v": (config["b"] - 1) ** 2}

            r2 = task2.optimize(n=8, verbose=False, brain_path=tmpdir)

            brain = Brain(path=tmpdir)
            self.assertEqual(brain.theory["total_tasks"], 2)
            self.assertIn("domain_a", brain.theory["domains_seen"])
            self.assertIn("domain_b", brain.theory["domains_seen"])
            self.assertEqual(brain.theory["total_experiments"], 16)
        finally:
            shutil.rmtree(tmpdir)

    def test_position_scores_accumulate(self):
        tmpdir = tempfile.mkdtemp()
        try:
            @theorist.experiment(
                search_space={"x": [1, 5, 10]},
                metric="loss",
                domain="test_pos",
            )
            def fn(config):
                return {"loss": abs(config["x"] - 5)}

            fn.optimize(n=6, verbose=False, brain_path=tmpdir)

            brain = Brain(path=tmpdir)
            scores = brain.theory["param_position_scores"]
            self.assertTrue(len(scores) > 0)
        finally:
            shutil.rmtree(tmpdir)


class TestTheoristDirect(unittest.TestCase):
    def test_direct_optimize(self):
        tmpdir = tempfile.mkdtemp()
        try:
            t = theorist.Theorist(domain="direct_test", brain_path=tmpdir)

            def fn(config):
                return {"loss": (config["x"] - 2) ** 2 + (config["y"] - 3) ** 2}

            results = t.optimize(
                fn=fn,
                search_space={"x": [0, 1, 2, 3, 4], "y": [0, 1, 2, 3, 4]},
                n=10,
                metric="loss",
                minimize=True,
                verbose=False,
            )
            self.assertIsInstance(results, Results)
            self.assertEqual(len(results.experiments), 10)
            self.assertLessEqual(results.best_metric, 10.0)
        finally:
            shutil.rmtree(tmpdir)

    def test_auto_baseline(self):
        tmpdir = tempfile.mkdtemp()
        try:
            t = theorist.Theorist(domain="auto_bl", brain_path=tmpdir)
            results = t.optimize(
                fn=lambda c: {"v": c["x"]},
                search_space={"x": [0, 1, 2, 3, 4]},
                n=3,
                metric="v",
                verbose=False,
            )
            # Auto baseline picks middle value (index 2 -> value 2)
            self.assertEqual(results.experiments[0].config["x"], 2)
        finally:
            shutil.rmtree(tmpdir)


class TestCompare(unittest.TestCase):
    def test_compare_returns_both_results(self):
        tmpdir = tempfile.mkdtemp()
        try:
            def simple_fn(config):
                return {"loss": (config["x"] - 2) ** 2}

            result = theorist.compare(
                fn=simple_fn,
                search_space={"x": [0, 1, 2, 3, 4]},
                n=10,
                metric="loss",
                minimize=True,
                brain_path=tmpdir,
                verbose=False,
            )
            self.assertIn("theorist", result)
            self.assertIn("random", result)
            self.assertIsInstance(result["theorist"], Results)
            self.assertIn("best_metric", result["random"])
            self.assertIn("best_config", result["random"])
            self.assertEqual(len(result["theorist"].experiments), 10)
            self.assertEqual(len(result["random"]["experiments"]), 10)
        finally:
            shutil.rmtree(tmpdir)

    def test_compare_maximize(self):
        tmpdir = tempfile.mkdtemp()
        try:
            def maximize_fn(config):
                return {"score": config["x"] * 10}

            result = theorist.compare(
                fn=maximize_fn,
                search_space={"x": [0, 1, 2, 3]},
                n=8,
                metric="score",
                minimize=False,
                brain_path=tmpdir,
                verbose=False,
            )
            self.assertGreaterEqual(result["theorist"].best_metric, 0)
            self.assertGreaterEqual(result["random"]["best_metric"], 0)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
