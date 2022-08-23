import numpy as np
import win32api
import win32gui
import win32con

from time import sleep


class VirtualKeyboard:
    def __init__(self, window_name, wait=0) -> None:
        # self.msg_func = win32api.SendMessage
        self.msg_func = win32api.PostMessage
        self.w_handle = win32gui.FindWindow(None, window_name)
        if self.w_handle == 0:
            if wait > 0:
                while self.w_handle == 0:
                    self.w_handle = win32gui.FindWindow(None, window_name)
                    sleep(wait)
            else:
                raise Exception(
                    f"Could not retrieve handle to the window {window_name}"
                )

        keys = {
            "DOWN": win32con.VK_DOWN,
            "UP": win32con.VK_UP,
            "LEFT": win32con.VK_LEFT,
            "RIGHT": win32con.VK_RIGHT,
            "Q": 0x51,
            "S": 0x53,
            "D": 0x44,
            "Z": 0x5A,
        }
        self.keys = keys
        self.keys_state = {k: False for k in keys.values()}

    def press_key(self, keycode):
        self.msg_func(self.w_handle, win32con.WM_KEYDOWN, keycode, 0)
        self.keys_state[keycode] = True

    def release_key(self, keycode):
        self.msg_func(self.w_handle, win32con.WM_KEYUP, keycode, 0)
        self.keys_state[keycode] = False

    def get_null_action(self):
        return np.zeros(len(self.keys))

    def update(self, action):
        assert len(action) == len(self.keys)
        for a, k in zip(action, self.keys_state):
            # if button needs to be pressed but not pressed
            if a and not self.keys_state[k]:
                self.press_key(k)
            # if button needs to be released but is pressed
            if not a and self.keys_state[k]:
                self.release_key(k)


if __name__ == "__main__":
    controller = VirtualKeyboard("Binding of Isaac: Repentance")
    action = [True, True, False, False, False, False, False, False]
    controller.update(action)
    sleep(5)
    action = [False, False, False, False, False, False, False, False]
    controller.update(action)
