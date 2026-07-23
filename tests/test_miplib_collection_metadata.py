import tempfile
import unittest
from pathlib import Path

from scip_cut_trace_v2.miplib_collection_metadata import build_metadata


HTML = """
<table id="miplibtable"><thead><tr><th>Instance</th></tr></thead><tbody>
<tr><td><a href="a">a</a></td><td>1</td><td>1</td><td>0</td><td>0</td>
<td>1</td><td>1</td><td>Alice</td><td>family-a</td><td>easy</td><td>1</td></tr>
<tr><td><a href="b">b</a></td><td>2</td><td>2</td><td>0</td><td>0</td>
<td>2</td><td>2</td><td>Bob</td><td>–</td><td>easy</td><td>Infeasible</td></tr>
<tr><td><a href="c">c</a></td><td>3</td><td>3</td><td>0</td><td>0</td>
<td>3</td><td>3</td><td>Carol</td><td>family-c</td><td>easy</td><td>3</td></tr>
</tbody></table>
"""


class MiplibCollectionMetadataTest(unittest.TestCase):
    def test_current_status_files_override_the_html(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            html = root / "collection.html"
            easy = root / "easy.test"
            infeasible = root / "infeasible.test"
            html.write_text(HTML, encoding="utf-8")
            easy.write_text("a.mps.gz\nb.mps.gz\n", encoding="utf-8")
            infeasible.write_text("b.mps.gz\n", encoding="utf-8")

            rows, provenance = build_metadata(html, easy, infeasible)

            self.assertEqual([row["status"] for row in rows], ["easy", "infeasible", "not-easy-current"])
            self.assertEqual(rows[1]["group"], "")
            self.assertEqual(rows[1]["tags"], "infeasible")
            self.assertEqual(provenance["summary"]["instances"], 3)


if __name__ == "__main__":
    unittest.main()
