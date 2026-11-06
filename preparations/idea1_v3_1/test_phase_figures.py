"""Hand-counted synthetic CPU checks; never reads the live reports."""
import csv
from pathlib import Path
import tempfile
import unittest

from .phase_figures import CAPTION,PHASES,build,pages,prepare


def quality(cid="pi0_mask",denominator=4,successes=1,expected=4,budget=3,scope="screen_init0"):
    return dict(condition=cid,model="pi0",scope=scope,suite="ALL",budget=budget,
        denominator=denominator,successes=successes,expected=expected)


def phase(name,count,failed=3,cid="pi0_mask",scope="screen_init0"):
    return dict(condition=cid,scope=scope,suite="ALL",progress_phase=name,count=count,
        failed_episode_denominator=failed,proportion=count/failed)


class PhaseFigureTests(unittest.TestCase):
    def test_hand_known_counts_proportions_and_safe_extra_phase(self):
        rows,order = prepare([quality()],[phase("approach",2),phase("other_observed",1)])
        self.assertEqual(order[:8],list(PHASES))
        self.assertEqual(order[-1],"other_observed")
        row = rows[0]
        self.assertEqual(row["counts"][order.index("approach")],2)
        self.assertAlmostEqual(row["proportions"][order.index("approach")],2/3)
        self.assertEqual(sum(row["counts"]),3)
        self.assertIn("NOT cause of failure",CAPTION)

    def test_zero_failure_missing_partial_are_distinct(self):
        rows,_ = prepare([quality("pi0_full",4,4,budget=0),quality("pi0_missing",0,0),
            quality("pi0_partial",2,2)],[])
        self.assertEqual([r["status"] for r in rows],["complete","missing","partial"])
        self.assertEqual([r["proportions"] for r in rows],[None,None,None])
        self.assertEqual([r["failures"] for r in rows],[0,0,0])
        for _,_,_,_,group in pages(rows):
            self.assertNotIn("pi0_missing",[r["condition"] for r in group])
        missing_full,_ = prepare([quality("pi0_full",0,0,budget=0),quality("pi0_partial",2,2)],[])
        candidate = next(p for p in pages(missing_full) if p[2]==3)
        self.assertEqual(candidate[4][0]["status"],"missing")
        self.assertIsNone(candidate[4][0]["proportions"])

    def test_failure_denominator_mismatch_missing_phase_rejected(self):
        with self.assertRaises(AssertionError):
            prepare([quality()],[phase("contact",2)])
        with self.assertRaises(AssertionError):
            prepare([quality()],[phase("contact",3,failed=4)])

    def test_bounded_pages_always_include_full_reference(self):
        raw = [quality("pi0_full",4,4,budget=0)]+[quality(f"pi0_variant{i}",2,2) for i in range(23)]
        rows,_ = prepare(raw,[])
        candidates = [p for p in pages(rows,10) if p[2]==3]
        self.assertEqual(len(candidates),3)
        for _,_,_,_,group in candidates:
            self.assertLessEqual(len(group),10)
            self.assertEqual(group[0]["condition"],"pi0_full")

    def test_cpu_render_new_directory_and_hashed_index(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)/"report"
            report.mkdir()
            for scope,name in (("screen_init0","quality_screen_init0.csv"),("combined_init0_4","quality_combined_init0_4.csv")):
                rows = [quality("pi0_full",4,4,budget=0,scope=scope),quality(scope=scope)]
                with (report/name).open("x",newline="") as stream:
                    writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            phases = [phase("contact",3,scope=scope) for scope in ("screen_init0","combined_init0_4")]
            with (report/"failure_progress_phases.csv").open("x",newline="") as stream:
                writer = csv.DictWriter(stream,fieldnames=list(phases[0]))
                writer.writeheader()
                writer.writerows(phases)
            output = Path(directory)/"plots"
            result = build(report,output)
            self.assertEqual(len(result["plots"]),4)
            self.assertEqual(len(result["csv_sources_sha256"]),3)
            self.assertTrue((output/"plot_index.json").is_file())
            self.assertTrue(all((output/r["path"]).is_file() for r in result["plots"]))
            with self.assertRaises(FileExistsError):
                build(report,output)


if __name__=="__main__":
    unittest.main()
