@echo off
REM Launched by Windows Task Scheduler so the 6-hour documented run is
REM independent of any terminal, console window or job object. Closing the
REM terminal that created the task does not touch this.
REM
REM Unlike the 12-hour run this supersedes, [strategy] kind = "baseline" makes
REM no model calls at all, so this needs no ANTHROPIC_* credentials and costs
REM nothing. The .env is still loaded by config for the author-hash salt.
cd /d "C:\Users\abkhalid\Documents\memecoin-trader"
"C:\Users\abkhalid\AppData\Local\hermes\bin\uv.exe" run python tools\document_run.py --hours 6 --out "runs\20260920T224616Z" > "runs\20260920T224616Z\console.log" 2>&1
