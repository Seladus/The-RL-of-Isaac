import subprocess
import warnings

from os.path import exists

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_speed_hack(window_name, value=None):
    print(window_name)
    return subprocess.Popen(
        [
            "speedhack/Dll Injector.exe",
            f"speedhack/Dll{value if value else '1'}.dll",
            window_name,
        ]
    )


if __name__ == "__main__":
    run_speed_hack()
