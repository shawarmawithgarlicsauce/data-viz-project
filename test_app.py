from streamlit.testing.v1 import AppTest


def test_app_runs_and_beats_baseline():
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    print(m)
    recall = float(m["Recall (failures caught)"].rstrip("%")) / 100
    assert recall > 0.7


if __name__ == "__main__":
    test_app_runs_and_beats_baseline()
