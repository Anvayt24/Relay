"""The demo page is served by the gateway at `/`, and never in the way of the API."""

import re
from pathlib import Path

import httpx

from relay.api.app import create_app
from relay.providers.mock import MockProvider

from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]


def _site(tmp_path: Path) -> Path:
    (tmp_path / "index.html").write_text("<!doctype html><title>page</title>", encoding="utf-8")
    (tmp_path / "style.css").write_text("body{}", encoding="utf-8")
    return tmp_path


async def _get(path: str, **settings: object) -> httpx.Response:
    app = create_app(relay_settings(**settings), providers={"mock": MockProvider()})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
        return await c.get(path)


async def test_the_page_is_served_at_the_root(tmp_path: Path) -> None:
    r = await _get("/", web_dir=str(_site(tmp_path)))
    assert r.status_code == 200 and "<title>page</title>" in r.text
    assert (await _get("/style.css", web_dir=str(_site(tmp_path)))).status_code == 200


async def test_the_page_never_shadows_the_api(tmp_path: Path) -> None:
    site = str(_site(tmp_path))
    assert (await _get("/healthz", web_dir=site)).json() == {"status": "ok"}
    missing = await _get("/v1/nope", web_dir=site)
    # An unknown API path is still Relay's JSON error, not the page and not a plain 404.
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "invalid_request"


async def test_ui_off_serves_no_page(tmp_path: Path) -> None:
    assert (await _get("/", web_dir=str(_site(tmp_path)), ui="off")).status_code == 404


async def test_a_missing_page_is_not_an_error(tmp_path: Path) -> None:
    # No index.html (a source checkout that was never built, or an API-only image).
    assert (await _get("/", web_dir=str(tmp_path / "nowhere"))).status_code == 404
    assert (await _get("/healthz", web_dir=str(tmp_path / "nowhere"))).status_code == 200


# ------------------------------------------------------------------ the page itself
def test_every_element_the_script_looks_up_exists_in_the_page() -> None:
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([\w-]+)"', html))
    looked_up: set[str] = set()
    for source in (ROOT / "web" / "src").glob("*.ts"):
        looked_up |= set(re.findall(r'\bel(?:<[^>]*>)?\("([\w-]+)"\)', source.read_text("utf-8")))
    for tab in ("playground", "chaos", "evidence"):  # looked up by template in main.ts
        looked_up |= {f"tab-{tab}", f"panel-{tab}"}
    assert looked_up - ids == set(), "the script reads ids the page does not have"


def test_the_page_loads_its_own_stylesheet_and_script() -> None:
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert 'href="style.css"' in html and (ROOT / "web" / "style.css").exists()
    assert 'src="dist/main.js"' in html and (ROOT / "web" / "src" / "main.ts").exists()
    code = "".join(p.read_text("utf-8") for p in (ROOT / "web" / "src").glob("*.ts"))
    assert not re.search(r"\.(innerHTML|outerHTML)|insertAdjacentHTML|document\.write", code), (
        "model output is untrusted: the page must build DOM nodes, never parse HTML"
    )
