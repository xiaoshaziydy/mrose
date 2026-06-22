import unittest
from pathlib import Path

from mrose_web import jobs


class WebJobHelpersTest(unittest.TestCase):
    def test_normalize_sequence_removes_whitespace_and_converts_u(self):
        self.assertEqual(jobs.normalize_sequence("acg u\n"), "ACGT")

    def test_invalid_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            jobs.normalize_sequence("ACGTX")

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


if __name__ == "__main__":
    unittest.main()
