"""Fill report_template.html with summary.json + findings/notes fragments -> report.html."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(HERE, "summarize.py")], check=True, stdout=subprocess.DEVNULL)
data = json.load(open(os.path.join(HERE, "summary.json")))
page = open(os.path.join(HERE, "report_template.html")).read()
for key, frag in (("__FINDINGS__", "findings.html"), ("__NOTES__", "notes.html")):
    p = os.path.join(HERE, frag)
    page = page.replace(key, open(p).read() if os.path.exists(p) else "")
page = page.replace("__DATA__", json.dumps({"summary": data["summary"], "names": data["names"]}))
open(os.path.join(HERE, "report.html"), "w").write(page)
print("wrote report.html", len(page), "bytes")
