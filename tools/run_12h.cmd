@echo off
REM Launched by Windows Task Scheduler so the 12-hour documented run is
REM independent of any terminal, console window or job object. Closing the
REM terminal that created the task does not touch this.
REM
REM Credentials come from .env. Claude Code injects ANTHROPIC_API_KEY and
REM ANTHROPIC_BASE_URL into its own child shells only, so a run launched any
REM other way saw no credentials and every model call failed with "Could not
REM resolve authentication method".
cd /d "C:\Users\abkhalid\Documents\memecoin-trader"
"C:\Users\abkhalid\AppData\Local\hermes\bin\uv.exe" run python tools\document_run.py --hours 12 --out "runs\20260920T024841Z" > "runs\20260920T024841Z\console.log" 2>&1
