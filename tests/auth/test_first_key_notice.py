"""The first-run banner is the one place a new operator sees their key, so it must read well and
its curl example must be copy-pasteable."""

from relay.auth.bootstrap import first_key_notice

KEY = "rk_TESTKEY123"


def test_the_key_appears_once_as_the_key_and_the_banner_says_it_is_shown_once() -> None:
    text = first_key_notice(KEY, "dev")
    assert "shown only once" in text
    assert text.count(f"\n   {KEY}\n") == 1  # on its own line, easy to select


def test_the_curl_example_is_real_multiline_shell() -> None:
    text = first_key_notice(KEY, "dev", base_url="http://localhost:9000")
    backslash = chr(92)
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.lstrip().startswith("curl "))
    # Two continuation lines, each ending in a backslash, then the final line of the command.
    assert lines[start].endswith(backslash) and lines[start + 1].endswith(backslash)
    assert not lines[start + 2].endswith(backslash)
    assert "http://localhost:9000/v1/chat/completions" in lines[start]
    assert f"Bearer {KEY}" in lines[start + 1]
    # No escape sequence left as literal text (backslash then the letter n).
    assert backslash + "n" not in text
