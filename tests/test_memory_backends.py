"""
Integration tests for MaintBot memory backends.

Runs the 6 standard test scenarios against a live deployment
and validates tool calls, memory operations, and response quality.

Usage:
    # Test against deployed app (requires DATABRICKS_CONFIG_PROFILE set)
    uv run python tests/test_memory_backends.py --app-url https://<your-app-url>

    # Test against local server
    uv run python tests/test_memory_backends.py --app-url http://localhost:8000
"""

import argparse
import json
import logging
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_token(profile: str = "stable-trvrmk") -> str:
    """Get Databricks OAuth token via CLI."""
    import subprocess
    result = subprocess.run(
        ["databricks", "auth", "token", "--profile", profile],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)["access_token"]


def send_message(
    app_url: str,
    token: str,
    message: str,
    thread_id: str,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Send a message to the app and return the full response."""
    resp = requests.post(
        f"{app_url}/invocations",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "input": [{"role": "user", "content": message}],
            "custom_inputs": {"thread_id": thread_id},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def extract_tool_calls(response: Dict) -> List[Dict]:
    """Extract tool calls from response output."""
    tools = []
    for item in response.get("output", []):
        if item.get("type") == "function_call":
            tools.append({"name": item["name"], "arguments": item.get("arguments", "")})
    return tools


def extract_tool_results(response: Dict) -> List[Dict]:
    """Extract tool results from response output."""
    results = []
    for item in response.get("output", []):
        if item.get("type") == "function_call_output":
            results.append({"output": item.get("output", "")})
    return results


def extract_response_text(response: Dict) -> str:
    """Extract the assistant's text response."""
    texts = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    texts.append(c["text"])
    return "\n".join(texts)


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.checks: List[Dict] = []
        self.elapsed_ms = 0
        self.error: Optional[str] = None

    def check(self, description: str, condition: bool, detail: str = ""):
        self.checks.append({"desc": description, "pass": condition, "detail": detail})
        if not condition:
            self.passed = False

    def finalize(self):
        self.passed = all(c["pass"] for c in self.checks)


def run_tests(app_url: str, token: str) -> List[TestResult]:
    """Run all 6 test scenarios."""
    results = []
    session = str(uuid.uuid4())[:8]
    thread_1 = f"test-{session}-001"
    thread_2 = f"test-{session}-002"
    thread_3 = f"test-{session}-003"
    thread_4 = f"test-{session}-004"

    # ── Test 1: Equipment Diagnostic ──
    t = TestResult("Q1: Equipment Diagnostic (new thread)")
    logger.info(f"Running {t.name}...")
    t0 = time.perf_counter()
    try:
        r = send_message(app_url, token,
            "The CNC-500X spindle is making a grinding noise during heavy cuts "
            "and we measured the motor temperature at 185F. What should I check first?",
            thread_1)
        t.elapsed_ms = (time.perf_counter() - t0) * 1000

        tools = extract_tool_calls(r)
        tool_names = [tc["name"] for tc in tools]
        text = extract_response_text(r)

        t.check("detect_equipment called", "detect_equipment" in tool_names)
        t.check("assess_severity called", "assess_severity" in tool_names)
        t.check("get_maintenance_memory called", "get_maintenance_memory" in tool_names)
        t.check("Response mentions spindle or bearing", "spindle" in text.lower() or "bearing" in text.lower())
        t.check("Response is substantial (>200 chars)", len(text) > 200, f"len={len(text)}")
    except Exception as e:
        t.error = str(e)
    t.finalize()
    results.append(t)

    # ── Test 2: Multi-Turn Follow-Up ──
    t = TestResult("Q2: Multi-Turn Follow-Up (same thread)")
    logger.info(f"Running {t.name}...")
    t0 = time.perf_counter()
    try:
        r = send_message(app_url, token,
            "Yes there is definite radial play. About 0.003 inches. "
            "The bearing preload seems off. What should I do?",
            thread_1)
        t.elapsed_ms = (time.perf_counter() - t0) * 1000

        text = extract_response_text(r)
        t.check("References CNC-500X or spindle (short-term memory)",
                "cnc" in text.lower() or "spindle" in text.lower() or "500x" in text.lower())
        t.check("Discusses bearing replacement", "bearing" in text.lower())
        t.check("Mentions 0.003 or play", "0.003" in text or "play" in text.lower())
    except Exception as e:
        t.error = str(e)
    t.finalize()
    results.append(t)

    # ── Test 3: Save to Memory ──
    t = TestResult("Q3: Save Learnings to Long-Term Memory")
    logger.info(f"Running {t.name}...")
    t0 = time.perf_counter()
    try:
        r = send_message(app_url, token,
            "We ended up replacing the bearings in-house. Took about 5 hours. "
            "Key learning: always check bearing preload first before replacing the whole spindle unit. "
            "Also the firmware v3.2.1 was causing false temperature alarms, so we updated to v3.2.3. "
            "Please save all of this to memory.",
            thread_1)
        t.elapsed_ms = (time.perf_counter() - t0) * 1000

        tools = extract_tool_calls(r)
        tool_names = [tc["name"] for tc in tools]
        tool_results = extract_tool_results(r)
        all_outputs = " ".join(tr["output"] for tr in tool_results)

        t.check("save_maintenance_memory called", "save_maintenance_memory" in tool_names)
        t.check("Memories extracted (>0 items)", "items found" in all_outputs and "0 items found" not in all_outputs,
                detail=all_outputs[:200])
        t.check("Memories stored (>0 stored)", "stored" in all_outputs and "0 new memories stored" not in all_outputs)
    except Exception as e:
        t.error = str(e)
    t.finalize()
    results.append(t)

    # Brief pause for VS eventual consistency
    time.sleep(2)

    # ── Test 4: Long-Term Recall (New Thread) ──
    t = TestResult("Q4: Long-Term Memory Recall (new thread)")
    logger.info(f"Running {t.name}...")
    t0 = time.perf_counter()
    try:
        r = send_message(app_url, token,
            "My CNC-500X spindle has a temperature alarm going off. "
            "The display shows 190F. Is this a real problem or a false alarm?",
            thread_2)
        t.elapsed_ms = (time.perf_counter() - t0) * 1000

        tools = extract_tool_calls(r)
        tool_names = [tc["name"] for tc in tools]
        tool_results = extract_tool_results(r)
        all_outputs = " ".join(tr["output"] for tr in tool_results)
        text = extract_response_text(r)

        t.check("get_maintenance_memory called", "get_maintenance_memory" in tool_names)
        t.check("Memory results mention firmware or 3.2.1",
                "firmware" in all_outputs.lower() or "3.2.1" in all_outputs,
                detail=all_outputs[:300])
        t.check("Response discusses false alarm possibility",
                "false alarm" in text.lower() or "firmware" in text.lower())
    except Exception as e:
        t.error = str(e)
    t.finalize()
    results.append(t)

    # ── Test 5: GDPR View ──
    t = TestResult("Q5: GDPR View All Memories")
    logger.info(f"Running {t.name}...")
    t0 = time.perf_counter()
    try:
        r = send_message(app_url, token,
            "Show me all the maintenance knowledge you have stored about me.",
            thread_3)
        t.elapsed_ms = (time.perf_counter() - t0) * 1000

        tools = extract_tool_calls(r)
        tool_names = [tc["name"] for tc in tools]
        tool_results = extract_tool_results(r)
        all_outputs = " ".join(tr["output"] for tr in tool_results)

        t.check("gdpr_view_memories called", "gdpr_view_memories" in tool_names)
        t.check("Returns non-empty memory list", "Found" in all_outputs and "0 memories" not in all_outputs,
                detail=all_outputs[:200])
    except Exception as e:
        t.error = str(e)
    t.finalize()
    results.append(t)

    # ── Test 6: Cross-Equipment Overview ──
    t = TestResult("Q6: Cross-Equipment Overview")
    logger.info(f"Running {t.name}...")
    t0 = time.perf_counter()
    try:
        r = send_message(app_url, token,
            "Give me a quick overview of all the equipment issues and tips "
            "you remember from our past conversations.",
            thread_4)
        t.elapsed_ms = (time.perf_counter() - t0) * 1000

        text = extract_response_text(r)
        t.check("Response mentions CNC-500X", "cnc" in text.lower() or "500x" in text.lower())
        t.check("Response is comprehensive (>400 chars)", len(text) > 400, f"len={len(text)}")
    except Exception as e:
        t.error = str(e)
    t.finalize()
    results.append(t)

    return results


def print_results(results: List[TestResult]):
    """Print formatted test results."""
    print("\n" + "=" * 70)
    print("MAINTBOT MEMORY BACKEND TEST RESULTS")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r.passed)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n{'[PASS]' if r.passed else '[FAIL]'} {r.name} ({r.elapsed_ms:.0f}ms)")
        if r.error:
            print(f"  ERROR: {r.error}")
        for check in r.checks:
            mark = "  [x]" if check["pass"] else "  [ ]"
            detail = f" — {check['detail']}" if check["detail"] else ""
            print(f"{mark} {check['desc']}{detail}")

    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All tests PASSED")
    else:
        print(f"FAILURES: {total - passed} test(s) failed")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test MaintBot memory backends")
    parser.add_argument("--app-url", required=True, help="App URL (e.g., https://maint-bot-app-xxx.aws.databricksapps.com)")
    parser.add_argument("--profile", default="stable-trvrmk", help="Databricks CLI profile for auth token")
    args = parser.parse_args()

    logger.info(f"Testing {args.app_url} with profile {args.profile}")
    token = get_token(args.profile)

    results = run_tests(args.app_url, token)
    print_results(results)

    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
