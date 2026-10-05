"""Explicit UNIT saved-schema journals; no checkpoint or completion artifact.

Numbers below exercise display semantics only. They are not training, medical
data, measured GPU activity, or evidence that an experiment completed.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hiercp_v1x.transition_learning_display import DLearningDisplay, JsonlTail, validation_line
from tools.watch_v17_d_learning import main


ROOT = Path(__file__).resolve().parents[1]


def validation(epoch, *, new_best=None, mrr=.2, hit=.125, recall=.0625):
    row = dict(UNIT_saved_schema_fixture=True, stage="validation", epoch=epoch,
        case_first_P_mrr=mrr, case_hit_at_1=hit, observed_micro_recall_at_1=recall,
        P_U_pair_win_rate=.6, P_U_softplus_loss=1.5)
    if new_best is not None:
        row["new_best"] = new_best
    return row


def update(step, *, epoch=1, loss=1.):
    return dict(UNIT_saved_schema_fixture=True, step=step, epoch=epoch, loss=loss,
        terms=dict(ranking_loss=.5, observation_ce=.3, alignment=.2),
        gradients=dict(L0=.8, L1=.7, L2=.6, L2_updates=.5),
        gradient_clip_norm=1.2, seconds=.4, peak_bytes=2*2**30,
        checkpoint="UNIT_UNUSED_REFERENCE_NEVER_CREATED_OR_OPENED")


def encoded(row):
    return json.dumps(row, allow_nan=False).encode("utf8") + b"\n"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SavedJournalUnit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="D_learning_display_UNIT_")
        self.root = Path(self.temp.name)
        self.training = self.root / "training"
        self.training.mkdir()
        self.curve = self.training / "curve.jsonl"
        self.updates = self.training / "updates.jsonl"

    def tearDown(self):
        if (self.root.resolve().parent != Path(tempfile.gettempdir()).resolve()
                or not self.root.name.startswith("D_learning_display_UNIT_")):
            raise RuntimeError("Refusing cleanup outside explicit UNIT temporary workspace")
        self.temp.cleanup()

    def write_history(self):
        rows = [validation(0), validation(1, new_best=True, mrr=.3, hit=.25, recall=.1),
                validation(2, new_best=False, mrr=.8, hit=.75, recall=.375)]
        rows.extend(validation(epoch, new_best=False, mrr=.8, hit=.75, recall=.375) for epoch in range(3, 6))
        rows[1]["epoch_wall_seconds"] = 120.
        self.curve.write_bytes(b"".join(encoded(row) for row in rows))
        self.updates.write_bytes(encoded(update(1, loss=2.)) + encoded(update(2, loss=1.)))
        return rows

    def test_all_saved_history_distinguishes_hit_and_micro_recall_and_saved_best_flag(self):
        rows = self.write_history()
        display = DLearningDisplay(self.root)
        lines, changed = display.poll()
        self.assertTrue(changed)
        self.assertEqual(len(lines), 6)
        self.assertEqual(display.history, {row["epoch"]: row for row in rows})
        self.assertIn("D INITIAL | MRR=0.200000 Hit@1=0.125000 R@1=0.062500", lines[0])
        self.assertIn("D epoch 01/40", lines[1])
        self.assertIn("Hit@1=0.250000 R@1=0.100000", lines[1])
        self.assertIn("NEW_BEST", lines[1])
        self.assertIn("wall=2.00min", lines[1])
        self.assertNotIn("NEW_BEST", lines[2])
        self.assertIn("MRR=0.800000", lines[2])
        self.assertIn("selected BEST epoch=1", lines[2])
        self.assertEqual(display.best_epoch, 1)  # Respect the saved flag, never infer argmax.
        self.assertEqual(display.poll(), ([], False))
        progress = display.progress_line()
        self.assertIn("epoch=1/40 step=2 loss=1.000000 avg2=1.500000", progress)
        self.assertIn("rank=0.50000 CE=0.30000 align=0.20000", progress)
        self.assertIn("module_grad=0.8/0.7/0.6/0.5", progress)
        self.assertIn("peak_GiB=2.00", progress)
        self.assertIn("MRR 0.200000 -> 0.800000 (delta=+0.600000)", display.delta_line())
        self.assertIn("Hit@1 0.125000 -> 0.750000", display.delta_line())
        self.assertIsNone(display.completion())

    def test_cli_prints_every_saved_epoch_and_creates_no_training_artifact(self):
        self.write_history()
        expected_paths = {str(self.curve), str(self.updates)}
        before = {path: digest(path) for path in expected_paths}
        runtime = ROOT / "hiercp_v1x/transition_runtime.py"
        runtime_before = digest(runtime)
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--experiment", str(self.root)])
        text = output.getvalue()
        self.assertEqual(text.count("D INITIAL |"), 1)
        for epoch in range(1, 6):
            self.assertEqual(text.count(f"D epoch {epoch:02d}/40 |"), 1)
        self.assertIn("Hit@1=case first rank is observed P; R@1=observed-P micro recall.", text)
        self.assertIn("selected BEST epoch=1", text)
        self.assertNotIn("D COMPLETE", text)
        self.assertEqual({str(path) for path in self.training.iterdir()}, expected_paths)
        self.assertEqual({path: digest(path) for path in expected_paths}, before)
        self.assertEqual(digest(runtime), runtime_before)

    def test_follow_displays_new_complete_epoch_on_next_poll_without_training_or_checkpoint(self):
        self.write_history()
        runtime = ROOT / "hiercp_v1x/transition_runtime.py"
        before = digest(runtime)
        calls = []
        def unit_writer_between_polls(seconds):
            calls.append(seconds)
            if len(calls) == 1:
                with self.curve.open("ab") as stream:
                    stream.write(encoded(validation(6, new_best=True, mrr=.4)))
            else:
                raise KeyboardInterrupt
        output = io.StringIO()
        with patch("tools.watch_v17_d_learning.time.sleep", side_effect=unit_writer_between_polls), redirect_stdout(output):
            main(["--experiment", str(self.root), "--follow"])
        text = output.getvalue()
        self.assertEqual(calls, [5, 5])
        self.assertEqual(text.count("D INITIAL |"), 1)
        for epoch in range(1, 7):
            self.assertEqual(text.count(f"D epoch {epoch:02d}/40 |"), 1)
        self.assertIn("D epoch 06/40 | MRR=0.400000", text)
        self.assertIn("selected BEST epoch=6", text)
        self.assertIn("Display stopped; D training and checkpoint are unchanged.", text)
        self.assertEqual(digest(runtime), before)
        self.assertEqual({path.name for path in self.training.iterdir()}, {"curve.jsonl", "updates.jsonl"})

    def test_fresh_process_poll_and_cli_import_no_torch_or_training_and_preserve_bytes(self):
        self.write_history()
        before = {str(path): digest(path) for path in (self.curve, self.updates,
                                                     ROOT / "hiercp_v1x/transition_runtime.py")}
        script = """
import contextlib, hashlib, io, json, pathlib, sys
from hiercp_v1x.transition_learning_display import DLearningDisplay
from tools.watch_v17_d_learning import main
experiment=pathlib.Path(sys.argv[1])
display=DLearningDisplay(experiment)
display.poll()
display.progress_line()
display.delta_line()
assert display.completion() is None
with contextlib.redirect_stdout(io.StringIO()):
    main(['--experiment',str(experiment)])
assert not any(name=='torch' or name.startswith('torch.') or name=='torch_geometric'
               or name.startswith('torch_geometric.') for name in sys.modules)
assert 'hiercp_v1x.transition_runtime' not in sys.modules
assert not list(experiment.rglob('*.pt'))
assert not list(experiment.rglob('training_complete.json'))
print(json.dumps({str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (experiment/'training/curve.jsonl',experiment/'training/updates.jsonl',
                               pathlib.Path(sys.argv[2]))},sort_keys=True))
"""
        result = subprocess.run([sys.executable, "-B", "-c", script, str(self.root),
            str(ROOT / "hiercp_v1x/transition_runtime.py")], cwd=ROOT, text=True,
            capture_output=True, check=False, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), before)
        self.assertEqual({path: digest(path) for path in before}, before)

    def test_partial_final_row_waits_then_appears_exactly_once(self):
        first, second = encoded(validation(0)), encoded(validation(1, new_best=True))
        cut = len(second)//2
        self.curve.write_bytes(first + second[:cut])
        tail = JsonlTail(self.curve)
        self.assertEqual(tail.read(), [validation(0)])
        self.assertTrue(tail.pending)
        self.assertEqual(tail.offset, len(first))
        self.assertEqual(tail.line, 1)
        self.assertEqual(tail.read(), [])
        self.assertTrue(tail.pending)
        with self.curve.open("ab") as stream:
            stream.write(second[cut:])
        self.assertEqual(tail.read(), [validation(1, new_best=True)])
        self.assertFalse(tail.pending)
        self.assertEqual(tail.line, 2)
        self.assertEqual(tail.read(), [])
        self.assertFalse(tail.pending)

    def test_whole_json_without_newline_is_pending_even_when_parseable(self):
        row = validation(0)
        self.curve.write_bytes(encoded(row)[:-1])
        display = DLearningDisplay(self.root)
        self.assertEqual(display.poll(), ([], False))
        self.assertTrue(display.curve.pending)
        self.assertEqual(display.history, {})
        with self.curve.open("ab") as stream:
            stream.write(b"\n")
        lines, changed = display.poll()
        self.assertEqual(len(lines), 1)
        self.assertFalse(changed)
        self.assertEqual(display.history, {0: row})
        self.assertEqual(display.poll(), ([], False))

    def test_truncated_replaced_and_disappeared_journals_are_refused(self):
        for change in ("truncate", "replace", "disappear"):
            with self.subTest(change=change):
                self.curve.write_bytes(encoded(validation(0)) + encoded(validation(1, new_best=True)))
                tail = JsonlTail(self.curve)
                self.assertEqual(len(tail.read()), 2)
                if change == "truncate":
                    self.curve.write_bytes(encoded(validation(0)))
                elif change == "replace":
                    replacement = self.training / "UNIT_replacement.jsonl"
                    replacement.write_bytes(encoded(validation(0)) + encoded(validation(1, new_best=True)))
                    os.replace(replacement, self.curve)
                else:
                    self.curve.unlink()
                with self.assertRaises(ValueError):
                    tail.read()

    def test_malformed_complete_rows_and_nonobjects_are_errors(self):
        for raw in (b'{"UNIT":}\n', b"\xff\n", b"[]\n", b"null\n"):
            with self.subTest(raw=raw):
                self.curve.write_bytes(raw)
                with self.assertRaises(ValueError):
                    JsonlTail(self.curve).read()

    def test_missing_journals_and_initial_only_do_not_fabricate_metrics(self):
        display = DLearningDisplay(self.root)
        self.assertEqual(display.poll(), ([], False))
        self.assertIn("NOT_YET_RECORDED", display.progress_line())
        self.assertIn("not yet available", display.delta_line())
        self.assertIsNone(display.completion())
        self.curve.write_bytes(encoded(validation(0)))
        lines, _ = display.poll()
        self.assertEqual(len(lines), 1)
        self.assertIn("not yet available", display.delta_line())
        self.assertIsNone(display.best_epoch)

    def test_invalid_validation_stage_cursor_best_flag_and_metrics_are_refused(self):
        changes = [{"stage": "UNKNOWN_UNIT_STAGE"}, {"epoch": True}, {"epoch": -1}, {"epoch": 41},
            {"new_best": "yes"}, {"case_first_P_mrr": float("nan")},
            {"case_hit_at_1": 1.1}, {"observed_micro_recall_at_1": True},
            {"P_U_pair_win_rate": -.1}, {"P_U_softplus_loss": -1.}]
        for values in changes:
            with self.subTest(values=values):
                row = validation(1, new_best=True)
                row.update(values)
                self.curve.write_bytes(json.dumps(row).encode() + b"\n")
                with self.assertRaises(ValueError):
                    DLearningDisplay(self.root).poll()
        row = validation(1, new_best=True)
        del row["case_hit_at_1"]
        with self.assertRaises(ValueError):
            validation_line(row)

    def test_duplicate_backwards_epochs_and_update_steps_are_refused(self):
        for second in (validation(0), validation(1, new_best=False)):
            with self.subTest(epoch=second["epoch"]):
                self.curve.write_bytes(encoded(validation(0)) + encoded(validation(1, new_best=True)) + encoded(second))
                with self.assertRaises(ValueError):
                    DLearningDisplay(self.root).poll()

        self.curve.write_bytes(encoded(validation(0)))
        for second in (update(1), update(0), update(2, epoch=0)):
            with self.subTest(update=second):
                self.updates.write_bytes(encoded(update(1)) + encoded(second))
                with self.assertRaises(ValueError):
                    DLearningDisplay(self.root).poll()

    def test_update_loss_and_optional_display_numbers_must_be_finite(self):
        for loss in (True, float("nan"), float("inf")):
            with self.subTest(loss=loss):
                self.updates.write_bytes(json.dumps(update(1, loss=loss)).encode() + b"\n")
                with self.assertRaises(ValueError):
                    DLearningDisplay(self.root).poll()
        self.updates.write_bytes(encoded(update(1)))
        display = DLearningDisplay(self.root)
        display.poll()
        display.recent[-1]["terms"]["ranking_loss"] = "UNIT_unknown_numeric_value"
        with self.assertRaises(ValueError):
            display.progress_line()

    def test_missing_saved_epoch_is_warned_without_synthetic_history_rows(self):
        self.curve.write_bytes(encoded(validation(0)) + encoded(validation(3, new_best=True)))
        display = DLearningDisplay(self.root)
        lines, _ = display.poll()
        self.assertIn("saved validation epochs 1..2 are missing; no metrics fabricated.", lines[1])
        self.assertEqual(set(display.history), {0, 3})

    def test_recent20_is_display_only_and_resets_at_epoch_boundary(self):
        self.curve.write_bytes(encoded(validation(0)) + encoded(validation(1, new_best=True)))
        self.updates.write_bytes(b"".join(encoded(update(step, loss=step/10)) for step in range(1, 27)))
        display = DLearningDisplay(self.root)
        lines, changed = display.poll()
        self.assertEqual(len(lines), 2)
        self.assertTrue(changed)
        self.assertEqual(display.last_step, 26)
        self.assertEqual(len(display.recent), 20)
        self.assertIn("avg20=1.650000", display.progress_line())
        self.assertEqual(set(display.history), {0, 1})
        with self.updates.open("ab") as stream:
            stream.write(encoded(update(27, epoch=2, loss=.25)))
        self.assertEqual(display.poll(), ([], True))
        self.assertEqual(len(display.recent), 1)
        self.assertIn("epoch=2/40 step=27 loss=0.250000 avg1=0.250000", display.progress_line())


if __name__ == "__main__":
    unittest.main()
