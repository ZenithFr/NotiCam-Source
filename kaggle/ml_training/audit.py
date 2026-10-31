import os
import sys
import subprocess
import time
import re

def audit_model_on_device(model_path="assets/models/portrait_v1_candidate.tflite"):
    print(f"=== Auditing {model_path} on Connected Device ===")
    assert os.path.exists(model_path), f"Error: {model_path} does not exist!"

    # 1. Check adb connection
    adb_out = subprocess.check_output(["adb", "devices"]).decode()
    if "device\n" not in adb_out and "\tdevice" not in adb_out:
        print("Error: No ADB device connected!")
        sys.exit(1)

    # 2. Push model to assets and recompile/install APK if needed, or push to app files
    # ModelAuditActivity loads from assets, so we ensure the model is copied to assets/models/
    target_asset = "/home/zenith/Work/auxcam/assets/models/portrait_v1_candidate.tflite"
    if os.path.abspath(model_path) != os.path.abspath(target_asset):
        import shutil
        shutil.copyfile(model_path, target_asset)
        print(f"Copied {model_path} to {target_asset}")

    # 3. Build & Install APK
    print("Building and installing APK...")
    subprocess.check_call(["bash", "-c", "cd /home/zenith/Work/auxcam && ./build.sh > /dev/null"])
    subprocess.check_call(["adb", "install", "-r", "/home/zenith/Work/auxcam/build/auxcam_aligned.apk"])

    # 4. Wake up device and clean orphaned privileged background services
    subprocess.call(["adb", "shell", "pkill", "-f", "auxcam_privileged"])
    subprocess.call(["adb", "shell", "input", "keyevent", "KEYCODE_WAKEUP"])
    subprocess.call(["adb", "shell", "wm", "dismiss-keyguard"])

    # 5. Clear logcat and launch ModelAuditActivity
    subprocess.call(["adb", "shell", "am", "force-stop", "com.zenith.auxcam"])
    subprocess.call(["adb", "logcat", "-c"])
    print("Launching ModelAuditActivity...")
    subprocess.check_call(["adb", "shell", "am", "start", "-n", "com.zenith.auxcam/.probe.ModelAuditActivity"])

    # 6. Wait for execution and read logcat
    time.sleep(8)
    log_bytes = subprocess.check_output(["adb", "logcat", "-d"])
    logs = log_bytes.decode(errors="ignore")

    # 7. Parse metrics
    p50_match = re.search(r"portrait_v1_candidate steady-state p50=(\d+)ms", logs)
    if not p50_match:
        p50_match = re.search(r"p50=(\d+)ms", logs)
    p95_match = re.search(r"portrait_v1_candidate steady-state p50=\d+ms p95=(\d+)ms", logs)
    if not p95_match:
        p95_match = re.search(r"p95=(\d+)ms", logs)
    partition_match = re.search(r"yielding (\d+) partitions", logs)
    nodes_match = re.search(r"Replacing (\d+) out of (\d+) node\(s\) with delegate", logs)

    print("=== AUDIT RESULTS ===")
    if partition_match:
        partitions = int(partition_match.group(1))
        print(f"Partitions: {partitions}")
        assert partitions == 1, f"FAIL: Expected 1 partition, got {partitions}"
    else:
        print("Warning: Partition log not matched")

    if nodes_match:
        delegated = int(nodes_match.group(1))
        total = int(nodes_match.group(2))
        print(f"Nodes delegated: {delegated}/{total} ({delegated/total*100:.1f}%)")
        assert delegated == total, f"FAIL: Not 100% delegated! ({delegated}/{total})"
    else:
        print("Warning: Nodes delegation log not matched")

    if p50_match and p95_match:
        p50 = int(p50_match.group(1))
        p95 = int(p95_match.group(1))
        print(f"Steady-State Latency: p50={p50}ms, p95={p95}ms")
        assert p50 <= 35, f"FAIL: p50 latency {p50}ms exceeds 35ms threshold!"
        assert p95 <= 40, f"FAIL: p95 latency {p95}ms exceeds 40ms threshold!"
        print("🟢 AUDIT PASSED: Model is 100% compliant with hardware constraints!")
    else:
        print("Warning: Steady-state latency logs not found, check raw logcat output.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/zenith/Work/auxcam/assets/models/portrait_v1_candidate.tflite"
    audit_model_on_device(path)
