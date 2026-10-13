"""Convert a pytest JUnit XML report (reports/test_results.xml) into a
standalone, readable HTML report (reports/test_report.html).

Written as a small stdlib-only script (no pytest-html dependency) so the
test report can always be regenerated offline.

Usage: python scripts/junit_to_html.py
"""
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "reports" / "test_results.xml"
HTML_PATH = ROOT / "reports" / "test_report.html"


def main():
    tree = ET.parse(XML_PATH)
    suite = tree.getroot().find("testsuite")
    if suite is None:
        suite = tree.getroot()  # some pytest versions emit <testsuite> as root

    total = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    passed = total - failures - errors - skipped
    duration = float(suite.get("time", 0))

    rows = []
    for case in suite.findall("testcase"):
        name = f"{case.get('classname')}::{case.get('name')}"
        case_time = float(case.get("time", 0))
        failure = case.find("failure")
        error = case.find("error")
        if failure is not None:
            status, detail, color = "FAILED", failure.get("message", ""), "#e34948"
        elif error is not None:
            status, detail, color = "ERROR", error.get("message", ""), "#e34948"
        else:
            status, detail, color = "PASSED", "", "#1baf7a"
        rows.append(
            f"<tr><td>{name}</td><td style='color:{color};font-weight:600'>{status}</td>"
            f"<td>{case_time:.3f}s</td><td>{detail}</td></tr>"
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>MMA3001 Occupancy Prediction - Test Report</title>
<style>
  body {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; background:#fcfcfb; color:#0b0b0b; margin:2rem; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .meta {{ color:#52514e; margin-bottom: 1.5rem; }}
  .summary {{ display:flex; gap:1rem; margin-bottom: 1.5rem; }}
  .stat {{ background:#f0efec; border-radius:8px; padding:0.75rem 1.25rem; }}
  .stat b {{ display:block; font-size:1.4rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align:left; padding:0.5rem 0.75rem; border-bottom:1px solid #e1e0d9; font-size:0.92rem; }}
  th {{ background:#f0efec; }}
</style>
</head>
<body>
  <h1>MMA3001 Occupancy Prediction &mdash; Test Report</h1>
  <p class="meta">Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
    from reports/test_results.xml (pytest JUnit output)</p>
  <div class="summary">
    <div class="stat"><b>{total}</b>Total</div>
    <div class="stat"><b style="color:#1baf7a">{passed}</b>Passed</div>
    <div class="stat"><b style="color:#e34948">{failures + errors}</b>Failed/Error</div>
    <div class="stat"><b>{skipped}</b>Skipped</div>
    <div class="stat"><b>{duration:.2f}s</b>Duration</div>
  </div>
  <table>
    <tr><th>Test</th><th>Status</th><th>Time</th><th>Detail</th></tr>
    {''.join(rows)}
  </table>
</body></html>"""

    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {HTML_PATH} ({passed}/{total} passed)")


if __name__ == "__main__":
    main()
