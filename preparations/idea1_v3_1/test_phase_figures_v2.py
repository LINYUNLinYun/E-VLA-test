"""Styling-only tests: unchanged scientific rows and readable six-layout labels."""
import csv
from pathlib import Path
import tempfile
import unittest

from . import phase_figures as core
from .phase_figures_v2 import build,labels,render
from .same_process_parity import sha
from .test_phase_figures import quality,phase


class PhaseFigureV2Tests(unittest.TestCase):
    def test_short_unique_labels_full_keys_reversible_and_no_overlap(self):
        names = ["pi0_full"]+[f"pi0_keep24_{i}_{layout}" for i,layout in enumerate(
            ("uniform","front_middle","back_middle","short_runs","seeded_random"),1)]
        raw = [quality(cid,40,34,40,budget=0 if cid=="pi0_full" else 3) for cid in names]
        phases = [phase("contact",6,6,cid=cid) for cid in names]
        rows,order = core.prepare(raw,phases)
        display = labels(rows,{})
        self.assertEqual([r["short_label"] for r in display],["Full [reference]","uniform","front-mid","back-mid","short-runs","random"])
        self.assertEqual([r["condition"] for r in display],names)
        figure = render(rows,order,display,"Synthetic six-layout fixture")
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for ax in figure.axes:
            boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
            self.assertTrue(all(not a.overlaps(b) for a,b in zip(boxes,boxes[1:])))
        import matplotlib.pyplot as plt
        plt.close(figure)
        self.assertEqual(rows,core.prepare(raw,phases)[0])

    def test_partial_missing_reference_and_duplicate_short_labels(self):
        rows,_ = core.prepare([quality("pi0_full",0,0,budget=0),quality("pi0_x_uniform",2,2),quality("pi0_y_uniform",2,2)],[])
        display = labels(rows,{})
        self.assertIn("MISSING | 0/4",display[0]["tick_label"])
        self.assertIn("F0 | 2/4 partial",display[1]["tick_label"])
        self.assertEqual([r["short_label"] for r in display][1:],["uniform","uniform [2]"])
        self.assertTrue(all(r["proportions"] is None for r in rows))

    def test_synthetic_build_hashes_frozen_core_and_same_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            report,output = Path(directory)/"report",Path(directory)/"v2"
            report.mkdir()
            raw = []
            for scope,name in (("screen_init0","quality_screen_init0.csv"),("combined_init0_4","quality_combined_init0_4.csv")):
                rows = [quality("pi0_full",4,4,budget=0,scope=scope),quality(scope=scope)]
                raw.extend(rows)
                with (report/name).open("x",newline="") as stream:
                    writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            phases = [phase("contact",3,scope=scope) for scope in ("screen_init0","combined_init0_4")]
            with (report/"failure_progress_phases.csv").open("x",newline="") as stream:
                writer = csv.DictWriter(stream,fieldnames=list(phases[0]))
                writer.writeheader()
                writer.writerows(phases)
            result = build(report,output)
            self.assertEqual(result["core_source_sha256"],sha(core.__file__))
            self.assertEqual(result["observations"],core.prepare(raw,phases)[0])
            self.assertEqual(len(result["plots"]),4)
            with self.assertRaises(FileExistsError):
                build(report,output)


if __name__=="__main__":
    unittest.main()
