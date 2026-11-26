"""Select and run a UI-S1 baseline on an OSWorld-Human 15–20 action task."""

import argparse
import json
import math
import os
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OSWORLD = ROOT.parent / "OSWorld"
EXAMPLES = OSWORLD / "evaluation_examples" / "examples"
KEY_ALIASES = {
    "control": "ctrl", "return": "enter", "escape": "esc",
    "61": "tab", "66": "enter", "67": "backspace", "111": "esc",
    "19": "up", "20": "down", "21": "left", "22": "right",
}
WEBDRIVER_KEYS = {
    "\ue003": "backspace", "\ue004": "tab", "\ue006": "enter", "\ue007": "enter",
    "\ue008": "shift", "\ue009": "ctrl", "\ue00a": "alt", "\ue00c": "esc",
    "\ue00d": "space", "\ue00e": "pageup", "\ue00f": "pagedown",
    "\ue010": "end", "\ue011": "home", "\ue012": "left", "\ue013": "up",
    "\ue014": "right", "\ue015": "down", "\ue017": "delete",
}
# Model output can contain the key character or its literal Unicode escape.
KEY_ALIASES.update(WEBDRIVER_KEYS)
KEY_ALIASES.update({f"\\u{ord(code):04x}": key for code, key in WEBDRIVER_KEYS.items()})
MODIFIERS = {"ctrl", "alt", "shift"}
KEYS = set(string.ascii_lowercase + string.digits) | {
    "enter", "tab", "esc", "space", "backspace", "delete", "up", "down",
    "left", "right", "home", "end", "pageup", "pagedown",
    *(f"f{number}" for number in range(1, 13)),
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def long_tasks(human_root):
    for domain in sorted(EXAMPLES.iterdir()):
        for annotation in sorted((human_root / domain.name).glob("*.json")):
            official = domain / annotation.name
            if not official.is_file():
                continue
            human = load_json(annotation)
            actions = human.get("human-ground-truth", {}).get("single-action")
            if isinstance(actions, list) and 15 <= len(actions) <= 20:
                task = load_json(official)
                if human.get("id") == task["id"]:
                    yield domain.name, task, len(actions)


def translate(action, size):
    kind = action["action"]
    if kind == "click":
        x, y = action["coordinate"]
        if type(x) is not int or type(y) is not int or not (0 <= x < size[0] and 0 <= y < size[1]):
            raise ValueError("Click is outside the observed screenshot.")
        return {"action_type": "CLICK", "x": x, "y": y}, 1
    if kind == "type":
        value = action["text"]
        if not isinstance(value, str) or not value:
            raise ValueError("Desktop typing currently supports printable ASCII.")
        value = value.replace("\r\n", "\n")
        text = value.rstrip("\n")
        enter_presses = len(value) - len(text)
        if any(not 32 <= ord(char) <= 126 for char in text):
            raise ValueError("Desktop typing currently supports printable ASCII.")
        if not text and enter_presses == 1:
            return {"action_type": "PRESS", "key": "enter"}, 1
        desktop = {"action_type": "TYPING", "text": text}
        if enter_presses == 1:
            desktop["press_enter"] = True
        elif enter_presses > 1:
            desktop["enter_presses"] = enter_presses
        return desktop, 1
    if kind == "key" or (kind == "system_button" and action["button"] in {"Enter", "Back"}):
        value = action["text"] if kind == "key" else {"Enter": "enter", "Back": "esc"}[action["button"]]
        keys = [key.strip().lower() for key in value.split("+")]
        keys = [KEY_ALIASES.get(key, key) for key in keys]
        if keys[-1] not in KEYS or any(key not in MODIFIERS for key in keys[:-1]):
            raise ValueError(f"Unsupported desktop key sequence: {value!r}")
        if len(keys) == 1:
            return {"action_type": "PRESS", "key": keys[0]}, 1
        return {"action_type": "HOTKEY", "keys": keys}, 1
    if kind == "wait":
        seconds = action["time"]
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Wait duration must be finite and nonnegative.")
        # OSWorld applies the pause twice for its WAIT action.
        return "WAIT", min(seconds, 10) / 2
    if kind == "terminate":
        return {"success": "DONE", "failure": "FAIL"}[action["status"]], 1
    raise ValueError(f"Unsupported desktop action: {kind}")


def run_trial(env, task, steps, device, output, history_mode="none", *, action_format="retry"):
    from PIL import Image

    observation = env.reset(task_config=task)
    history = []
    # Reprocessing a growing screenshot history can take much longer on CPU.
    inference_timeout = 3600 if history_mode == "full" else 900
    stop = "step_limit"
    executed = 0
    format_retries = 0
    attempts = (False, True) if action_format == "retry" else (False,)
    for number in range(1, steps + 1):
        step = output / f"step-{number:02}"
        step.mkdir()
        screen = step / "before.png"
        screen.write_bytes(observation["screenshot"])
        with Image.open(screen) as image:
            size = image.size
        print(f"Step {number}: running UI-S1", flush=True)
        prompt = task["instruction"] + "\nUse desktop keys such as Ctrl+A or Enter for key actions."
        command = [
            "bash", str(ROOT / "run.sh"), "--image", str(screen), "--goal", prompt,
            "--device", device, "--max-new-tokens", "160",
        ]
        if history_mode == "full":
            history_file = step / "history.json"
            history_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            command.extend(["--history-file", str(history_file)])
        inference_started = time.perf_counter()
        for retry in attempts:
            prefix = "retry-" if retry else ""
            raw_output = step / f"{prefix}raw-output.txt"
            attempt = [*command, "--raw-output", str(raw_output)]
            if retry:
                format_retries += 1
                print(f"Step {number}: retrying invalid model output with action-only formatting", flush=True)
            if retry or action_format == "constrained":
                attempt.append("--action-only")
            with (step / f"{prefix}inference.log").open("w") as stderr:
                prediction = subprocess.run(
                    attempt, stdout=subprocess.PIPE, stderr=stderr, timeout=inference_timeout,
                )
            (step / f"{prefix}prediction.json").write_bytes(prediction.stdout)
            if prediction.returncode != os.EX_DATAERR:
                break
        timings = {
            "inference_seconds": round(time.perf_counter() - inference_started, 3),
            "format_retry": retry,
        }
        timing_file = step / "timing.json"
        timing_file.write_text(json.dumps(timings, indent=2), encoding="utf-8")
        if prediction.returncode:
            stop = "inference_error"
            break
        try:
            action, pause = translate(json.loads(prediction.stdout)["screenshot_action"], size)
        except (KeyError, TypeError, ValueError) as error:
            (step / "action-error.txt").write_text(str(error) + "\n", encoding="utf-8")
            stop = "unsupported_or_invalid_action"
            break
        (step / "action.json").write_text(json.dumps(action) + "\n", encoding="utf-8")
        execution_started = time.perf_counter()
        observation, _, done, _ = env.step(action, pause)
        timings["vm_action_and_screenshot_seconds"] = round(time.perf_counter() - execution_started, 3)
        executed += 1
        (step / "after.png").write_bytes(observation["screenshot"])
        timing_file.write_text(json.dumps(timings, indent=2), encoding="utf-8")
        print(f"Step {number}: inference {timings['inference_seconds']:.2f}s, "
              f"VM action and screenshot {timings['vm_action_and_screenshot_seconds']:.2f}s", flush=True)
        if history_mode == "full":
            history.append({
                "screenshot": str(screen.relative_to(output)),
                "model_response": raw_output.read_text(encoding="utf-8"),
                "executed_action": action,
            })
        if done:
            stop = "model_reported_success" if action == "DONE" else "model_reported_failure"
            break
    return {
        "score": float(env.evaluate()), "stop": stop, "steps_executed": executed,
        "format_retries": format_retries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="List matched tasks and their human action counts.")
    run_parser = subparsers.add_parser("run", help="Run on a dedicated VMware benchmark VM.")
    for subparser in (list_parser, run_parser):
        subparser.add_argument("--human-root", type=Path, required=True)
    run_parser.add_argument("--task", required=True, metavar="TASK_ID")
    run_parser.add_argument("--vmx", type=Path, required=True, help="Dedicated VMware benchmark VM.")
    run_parser.add_argument("--snapshot", required=True, help="Snapshot OSWorld restores before task setup.")
    run_parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    run_parser.add_argument("--history", choices=("full", "none"), default="full",
                            help="Include every previous observation and action in each model turn (default: full).")
    run_parser.add_argument("--action-format", choices=("retry", "constrained"), default="retry",
                            help="Use the action constraint on retry or on the first and only attempt (default: retry).")
    args = parser.parse_args()
    if not args.human_root.is_dir():
        parser.error("--human-root must point to an OSWorld-Human checkout.")
    candidates = list(long_tasks(args.human_root))
    if not candidates:
        parser.error("No matching OSWorld tasks with 15–20 human actions were found.")
    if args.command == "list":
        for app, task, count in candidates:
            print(f"{count:2}  {app:<20}  {task['id']}  {' '.join(task['instruction'].split())}")
        return

    matches = [(app, task, count) for app, task, count in candidates if task["id"] == args.task]
    if len(matches) != 1:
        parser.error("--task must be one unique task shown by list.")
    vmx = args.vmx.resolve()
    if not vmx.is_file() or vmx.suffix != ".vmx":
        parser.error("--vmx must point to a VMware configuration file.")
    active_poster_vm = OSWORLD / "vmware_vm_data/Ubuntu-arm/Ubuntu.vmx"
    if vmx == active_poster_vm.resolve():
        parser.error("Use a dedicated VM. OSWorld will restore the specified snapshot.")
    app, task, reference_actions = matches[0]

    sys.path.insert(0, str(OSWORLD))
    os.chdir(OSWORLD)

    output = Path(tempfile.mkdtemp(prefix="osworld-ui-s1-baseline-"))
    print(f"Logs: {output}", flush=True)
    result = {
        "task": task["id"], "domain": app, "human_actions": reference_actions,
        "step_budget": 20, "history": args.history, "action_format": args.action_format,
    }
    try:
        from desktop_env.desktop_env import DesktopEnv

        env = DesktopEnv(provider_name="vmware", path_to_vm=str(vmx), snapshot_name=args.snapshot,
                         cache_dir=str(output / "cache"), action_space="computer_13",
                         require_a11y_tree=False)
        try:
            format_options = {"action_format": args.action_format} if args.action_format != "retry" else {}
            result.update(run_trial(env, task, 20, args.device, output, args.history, **format_options))
        finally:
            env.close()
    except Exception as error:
        result.update(score=None, stop="execution_error", error=str(error))
    summary = json.dumps(result, ensure_ascii=False, indent=2)
    (output / "result.json").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return int(result["score"] is None)


if __name__ == "__main__":
    raise SystemExit(main())
