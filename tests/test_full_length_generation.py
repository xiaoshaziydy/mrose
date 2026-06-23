import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from generation.full_length import generate_full_length


class FullLengthGenerationTest(unittest.TestCase):
    def test_three_utr_component_is_not_fixed_to_input_length(self):
        args = SimpleNamespace(
            output_dir=Path("/tmp/out"),
            num_samples=10,
            top_k=2,
            python="python",
            five_utr_script=Path("generation/5utr/generate_5utr.py"),
            five_utr_checkpoint=Path("generation/5utr/Model.pth"),
            five_utr_fasta=Path("generation/examples/5utr_template.fasta"),
            cds_script=Path("generation/cds/generate_cds.py"),
            cds_checkpoint=Path("generation/cds/Model.pth"),
            cds_fasta=Path("generation/examples/cds_template.fasta"),
            three_utr_script=Path("generation/3utr/generate_3utr.py"),
            three_utr_checkpoint=Path("generation/3utr/Model.pth"),
            three_utr_fasta=Path("generation/examples/3utr_template.fasta"),
            device="cpu",
            temperature=1.0,
            output_prefix="example",
            cds_mfe_weight=0.0,
            cds_batch_size=32,
        )

        commands = generate_full_length.build_commands(args)

        self.assertNotIn("--match_input_length", commands["3utr"])

    def test_merge_outputs_combines_same_rank_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "5utr").mkdir()
            (output_dir / "cds").mkdir()
            (output_dir / "3utr").mkdir()

            (output_dir / "5utr" / "example_5utr_top2.csv").write_text(
                "sequence,final_score\nAAAA,0.9\nCCCC,0.8\n"
            )
            (output_dir / "cds" / "example_cds_top.csv").write_text(
                "sequence,final_score\nATG,0.7\nGCC,0.6\n"
            )
            (output_dir / "3utr" / "example_3utr_top2.csv").write_text(
                "sequence,final_score\nTT,0.5\nGG,0.4\n"
            )

            Args = SimpleNamespace(top_k=2, output_dir=output_dir, output_prefix="example")
            rows = generate_full_length.merge_outputs(Args)
            self.assertEqual(rows[0]["sequence"], "AAAAATGTT")
            self.assertEqual(rows[1]["sequence"], "CCCCGCCGG")
            self.assertEqual(rows[0]["full_length"], 9)

            csv_path = output_dir / "full.csv"
            fasta_path = output_dir / "full.fasta"
            generate_full_length.write_csv(csv_path, rows)
            generate_full_length.write_fasta(fasta_path, rows)

            self.assertIn("AAAAATGTT", csv_path.read_text())
            self.assertIn(">full_length_rank_1", fasta_path.read_text())


if __name__ == "__main__":
    unittest.main()
