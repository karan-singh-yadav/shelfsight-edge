from shelfsight.system_info import collect_system_info, package_versions


def test_collect_system_info_has_benchmark_fields():
    info = collect_system_info()
    for key in ("os", "python", "cpu_model", "cpu_logical_cores", "ram_total_gb", "packages"):
        assert key in info
    assert info["ram_total_gb"] > 0
    assert info["packages"]["numpy"] != "not installed"


def test_missing_package_reported():
    assert package_versions(("definitely-not-a-real-package",)) == {
        "definitely-not-a-real-package": "not installed"
    }
