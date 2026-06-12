import os
import time
import glob

def get_latest_log() -> str:
    # Default to local cot_brain_debug.log if it exists
    default_log = "cot_brain_debug.log"
    if os.path.exists(default_log):
        return default_log
    
    # Or find any other log file in the current directory
    log_files = glob.glob("*.log")
    if not log_files:
        return None
    # Sort by modification time to find the newest one
    log_files.sort(key=os.path.getmtime)
    return log_files[-1]

def watch() -> None:
    log_file = get_latest_log()
    if not log_file:
        print("No task log files found in the brain directory.")
        return
    
    print(f"Streaming newest task log: {os.path.basename(log_file)}")
    print("=" * 60)
    
    # Open and stream the log
    with open(log_file, "r", encoding="utf-8") as f:
        # 1. Print existing lines
        print(f.read(), end="")
        
        # 2. Watch for new lines (tail -f implementation)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)  # Sleep briefly to avoid high CPU usage
                continue
            print(line, end="", flush=True)

if __name__ == "__main__":
    try:
        watch()
    except KeyboardInterrupt:
        print("\nStopped streaming log.")
