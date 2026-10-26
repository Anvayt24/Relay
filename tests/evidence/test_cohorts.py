from relay.evidence.cohorts import cohort_of, script_of


def test_script_detection() -> None:
    assert script_of("How are you today?") == "latin"
    assert script_of("आप कैसे हैं? मैं ठीक हूँ") == "devanagari"
    assert script_of("你好 今天天气怎么样") == "cjk"
    assert script_of("Привет, как дела?") == "cyrillic"
    assert script_of("12345 !!!") == "other"


def test_cohort_key_is_a_bounded_set_of_values() -> None:
    c = cohort_of("short question")
    assert c.key == "lang=latin,len=short,code=false,tools=false"
    assert cohort_of("x" * 500).length == "medium"
    assert cohort_of("x" * 5000).length == "long"
    assert cohort_of("see ```py\nprint(1)\n```").code is True
