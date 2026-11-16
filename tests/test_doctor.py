import json
import hashlib

from traffic_risk.doctor import diagnose, main


def test_missing_assets_and_dataset_have_remedies(tmp_path):
    result = diagnose({"model_dir": str(tmp_path), "dataset_root": str(tmp_path / "missing")})
    assert not result["ok"]
    failures = {c["name"]: c for c in result["checks"] if not c["ok"]}
    assert "yolo/yolo11n.pt" in failures
    assert "dataset" in failures
    assert all(c["remedy"] for c in failures.values())


def test_corrupt_manifest_and_invalid_mode(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{broken")
    result = diagnose({"mode": "wrong", "model_manifest": str(manifest)})
    assert not result["ok"]
    assert {c["name"] for c in result["checks"] if not c["ok"]} >= {"manifest", "mode"}


def test_main_rejects_non_object_config(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text("[]")
    assert main(["--config", str(config)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_models_requires_manifest(tmp_path):
    result = diagnose({"mode": "models", "model_dir": str(tmp_path)})
    assert not result["ok"]
    assert any(c["name"] == "manifest" and not c["ok"] for c in result["checks"])


def test_doctor_dispatch_does_not_load_runtime(monkeypatch):
    import traffic_risk.cli as cli
    monkeypatch.setattr("traffic_risk.doctor.main", lambda args: 7)
    assert cli.main(["doctor"]) == 7


def test_asset_checksum_is_checked(tmp_path):
    asset = tmp_path / "yolo" / "yolo11n.pt"
    asset.parent.mkdir()
    asset.write_bytes(b"fake")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"distribution_status": "public", "assets": [{
        "destination": "yolo/yolo11n.pt", "size": 4,
        "sha256": hashlib.sha256(b"fake").hexdigest(),
    }]}))
    config = {"model_manifest": str(manifest), "model_dir": str(tmp_path)}
    assert next(c for c in diagnose(config)["checks"] if c["name"] == "yolo/yolo11n.pt")["ok"]
    asset.write_bytes(b"bad!")
    assert not next(c for c in diagnose(config)["checks"] if c["name"] == "yolo/yolo11n.pt")["ok"]
