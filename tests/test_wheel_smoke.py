from tools.wheel_smoke import isolated_subprocess_environment


def test_isolated_subprocess_environment_filters_overrides_and_keeps_system_vars(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "D:/Codes/SafeSight/src")
    monkeypatch.setenv("pythonpath", "another/source/path")
    monkeypatch.setenv("RISK_UPLOAD_DIR", "D:/Codes/SafeSight/local/uploads")
    monkeypatch.setenv("risk_mode", "models")
    monkeypatch.setenv("PATH", "system/path")
    monkeypatch.setenv("SystemRoot", "C:/Windows")
    monkeypatch.setenv("TEMP", "system/temp")

    env = isolated_subprocess_environment()

    assert not any(key.upper() == "PYTHONPATH" for key in env)
    assert not any(key.upper().startswith("RISK_") for key in env)
    assert env["PATH"] == "system/path"
    assert env["TEMP"] == "system/temp"
    system_env = {key.upper(): value for key, value in env.items()}
    assert system_env["SYSTEMROOT"] == "C:/Windows"
