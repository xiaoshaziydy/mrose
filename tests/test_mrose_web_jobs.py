import unittest
import tempfile
from pathlib import Path

from mrose_web import jobs


class WebJobHelpersTest(unittest.TestCase):
    def test_normalize_sequence_removes_whitespace_and_converts_u(self):
        self.assertEqual(jobs.normalize_sequence("acg u\n"), "ACGT")

    def test_invalid_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            jobs.normalize_sequence("ACGTX")

    def test_write_json_round_trips_with_read_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_root = jobs.JOB_ROOT
            try:
                jobs.JOB_ROOT = Path(tmp)
                payload = {"job_id": "abc", "status": "queued"}
                jobs.write_json(jobs.status_path("abc"), payload)
                self.assertEqual(jobs.read_status("abc"), payload)
            finally:
                jobs.JOB_ROOT = original_root

    def test_build_command_uses_region_script_and_fasta(self):
        cmd = jobs.build_command(
            "5utr",
            Path("/tmp/input.fasta"),
            Path("/tmp/out"),
            num_samples=12,
            top_k=3,
            device="cpu",
            temperature=0.8,
            match_input_length=False,
        )
        self.assertIn(str(jobs.ROOT / "generation" / "5utr" / "generate_5utr.py"), cmd)
        self.assertIn("/tmp/input.fasta", cmd)
        self.assertIn("/tmp/out", cmd)
        self.assertIn("12", cmd)
        self.assertIn("3", cmd)
        self.assertIn("cpu", cmd)

    def test_five_utr_command_uses_two_sample_floor(self):
        cmd = jobs.build_command(
            "5utr",
            Path("/tmp/input.fasta"),
            Path("/tmp/out"),
            num_samples=1,
            top_k=1,
            device="cpu",
            temperature=1.0,
            match_input_length=False,
        )
        self.assertEqual(cmd[cmd.index("--num_samples") + 1], "2")

    def test_three_utr_match_length_flag_is_request_controlled(self):
        cmd = jobs.build_command(
            "3utr",
            Path("/tmp/input.fasta"),
            Path("/tmp/out"),
            num_samples=10,
            top_k=2,
            device="cpu",
            temperature=1.0,
            match_input_length=True,
        )
        self.assertIn("--match_input_length", cmd)

    def test_validate_full_request_returns_cleaned_parts(self):
        parts = jobs.validate_request(
            "full",
            "ACGT",
            num_samples=10,
            top_k=2,
            sequence_5utr="acgu",
            sequence_cds="augccc",
            sequence_3utr="uuu",
        )
        self.assertEqual(
            parts,
            {"5utr": "ACGT", "cds": "ATGCCC", "3utr": "TTT"},
        )

    def test_full_request_rejects_short_cds_part(self):
        with self.assertRaises(ValueError):
            jobs.validate_request(
                "full",
                "ACGT",
                num_samples=10,
                top_k=2,
                sequence_5utr="ACGT",
                sequence_cds="AT",
                sequence_3utr="TTTT",
            )

    def test_create_full_job_writes_real_fasta_lines(self):
        class FakeExecutor:
            def submit(self, *args, **kwargs):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            original_root = jobs.JOB_ROOT
            original_executor = jobs.EXECUTOR
            try:
                jobs.JOB_ROOT = Path(tmp)
                jobs.EXECUTOR = FakeExecutor()
                status = jobs.create_job(
                    region="full",
                    sequence="ACGTATGTTT",
                    sequence_5utr="ACGT",
                    sequence_cds="ATG",
                    sequence_3utr="TTT",
                    num_samples=1,
                    top_k=1,
                    device="cpu",
                    temperature=1.0,
                    match_input_length=True,
                )
                job_dir = Path(tmp) / status["job_id"]
                self.assertEqual(
                    (job_dir / "input_5utr.fasta").read_text().splitlines(),
                    [">mrose_input_5utr", "ACGT"],
                )
                self.assertEqual(
                    (job_dir / "input_cds.fasta").read_text().splitlines(),
                    [">mrose_input_cds", "ATG"],
                )
            finally:
                jobs.JOB_ROOT = original_root
                jobs.EXECUTOR = original_executor

    def test_merge_full_mrna_outputs_combines_ranked_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for region in ("5utr", "cds", "3utr"):
                (root / region).mkdir()
            (root / "5utr" / "mrose_5utr_top2.csv").write_text(
                "sequence,final_score\nAAAA,0.9\nCCCC,0.8\n"
            )
            (root / "cds" / "mrose_cds_top.csv").write_text(
                "sequence,final_score\nATG,0.7\nGCC,0.6\n"
            )
            (root / "3utr" / "mrose_3utr_top2.csv").write_text(
                "sequence,final_score\nTT,0.5\nGG,0.4\n"
            )

            jobs.merge_full_mrna_outputs(root, top_k=2)

            csv_text = (root / "mrose_full_top.csv").read_text()
            fasta_text = (root / "mrose_full_top.fasta").read_text()
            self.assertIn("AAAAATGTT", csv_text)
            self.assertIn("CCCCGCCGG", csv_text)
            self.assertIn(">mrose_full_rank_1", fasta_text)
            self.assertIn("AAAAATGTT", fasta_text)


if __name__ == "__main__":
    unittest.main()
