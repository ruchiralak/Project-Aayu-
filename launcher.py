import subprocess

try:
    # Use the absolute path to app.py here:
    app_path = r"C:\health_predicitor_framework\app.py"

    subprocess.run(
        ["streamlit", "run", app_path],
        check=True,
        capture_output=True,
        text=True
    )
except subprocess.CalledProcessError as e:
    with open("error_log.txt", "w", encoding="utf-8") as f:
        f.write("STREAMLIT FAILED TO START\n")
        f.write(f"Return Code: {e.returncode}\n\n")
        f.write("STDOUT:\n" + e.stdout + "\n")
        f.write("STDERR:\n" + e.stderr + "\n")
