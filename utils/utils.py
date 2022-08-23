import subprocess
import os
from time import sleep
import win32gui


def run_game(exe_path, args):
    p = subprocess.Popen([exe_path] + args)
    return p


def kill_game():
    os.system('taskkill /F /IM "isaac-ng.exe" /T')


def kill_process(window_name):
    os.system(f'taskkill /F /FI "WindowTitle eq {window_name}"')


def kill_steam():
    os.system("taskkill /F /IM Steam.exe")


def changeWindowName(current_name, new_name):
    handle = win32gui.FindWindow(None, current_name)
    win32gui.SetWindowText(handle, new_name)
