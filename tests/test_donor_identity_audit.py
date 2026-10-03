import json

from sca3_compass import repository


def test_legacy_identity_annotation_does_not_rewrite_historical_file(tmp_path, monkeypatch):
    monkeypatch.setattr(repository, "PROJECT_ROOT", tmp_path)
    folder = tmp_path / "data/processed"
    folder.mkdir(parents=True)
    original = {"limitations": ["Eight independent donors sharply limit precision."], "metrics": {"donors": 8}}
    for filename, loader in (
        ("gse309548_analysis.json", repository.load_transcriptomics_analysis),
        ("gse309548_gene_analysis.json", repository.load_gene_expression_analysis),
    ):
        path = folder / filename
        path.write_text(json.dumps(original), encoding="utf-8")
        annotated = loader()
        assert annotated["identity_audit"]["status"] == "unresolved"
        assert "Eight independent donors" not in " ".join(annotated["limitations"])
        assert annotated["metrics"]["donors"] == 8
        assert json.loads(path.read_text()) == original
