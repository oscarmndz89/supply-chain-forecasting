"""
Headless test for streamlit_app.py using Streamlit's AppTest framework.
Runs each of the 4 pages and asserts no exceptions are raised.
"""
import io
import sys
from pathlib import Path

# Force UTF-8 output on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Make sure project root is on the path (needed if src/ imports appear)
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parent / "streamlit_app.py"

PAGES = [
    "📊  Carrier Risk Scorecard",
    "⏱  Delay Forecast",
    "💸  Cost Variance Alerts",
    "🗺  Lane Intelligence",
]


def test_page(page_name: str) -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    if page_name != PAGES[0]:
        # Navigate to target page via sidebar radio
        at.sidebar.radio[0].set_value(page_name).run()

    if at.exception:
        raise RuntimeError(
            f"Page '{page_name}' raised an exception:\n{at.exception}"
        )
    print(f"  PASS  {page_name}")


if __name__ == "__main__":
    print(f"Testing {APP_PATH.name} ...")
    print()
    errors = []
    for page in PAGES:
        try:
            test_page(page)
        except Exception as exc:
            errors.append((page, exc))
            print(f"  FAIL  {page}")
            print(f"        {exc}")

    print()
    if errors:
        print(f"{len(errors)} page(s) failed.")
        sys.exit(1)
    else:
        print(f"All {len(PAGES)} pages passed — zero exceptions.")
