import os
import subprocess
import sys
from tqdm import tqdm


# credit: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running (tokland's answer)
def run_command(script_name, **kwargs):
    """Run a command while printing the live output"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"Starting {script_name}...")
    command = [sys.executable, script_path]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    while True:  # Could be more pythonic with := in Python3.8+
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        print(line.decode(), end="")
    print(f"{script_name} finished successfully.")


def main():
    scripts = [
        "download_samformer_dataset.py",
        "download_cifar10_dataset.py",
    ]
    for script in scripts:
        run_command(script)


if __name__ == "__main__":
    main()
