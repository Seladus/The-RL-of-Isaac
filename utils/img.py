from time import sleep
import win32gui
import win32ui
import numpy as np
from ctypes import windll
from PIL import Image


class ImageCapture:
    def __init__(self, window_name) -> None:
        self.hwnd = win32gui.FindWindow(None, window_name)
        # self.hwndDC = win32gui.GetWindowDC(self.hwnd)
        # # create rendering context
        # self.mfcDC = win32ui.CreateDCFromHandle(self.hwndDC)

    def capture_frame(self):
        left, top, right, bot = win32gui.GetClientRect(self.hwnd)

        # left, top, right, bot = win32gui.GetWindowRect(hwnd)
        w = right - left
        h = bot - top

        self.hwndDC = win32gui.GetWindowDC(self.hwnd)
        # create rendering context
        self.mfcDC = win32ui.CreateDCFromHandle(self.hwndDC)

        self.saveDC = self.mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(self.mfcDC, w, h)

        # print("Sleect object")
        self.saveDC.SelectObject(saveBitMap)
        # print("Print windows")
        result = windll.user32.PrintWindow(self.hwnd, self.saveDC.GetSafeHdc(), 1)

        # print("Get infos")
        bmpinfo = saveBitMap.GetInfo()
        # print("get map bits")
        bmpstr = saveBitMap.GetBitmapBits(True)

        # print("Image from buffer")
        im = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
            0,
            1,
        )

        # print("Delete bitmap")
        win32gui.DeleteObject(saveBitMap.GetHandle())
        # print("deleting saveDC")
        self.saveDC.DeleteDC()
        # print("deleting mfcDC")
        self.mfcDC.DeleteDC()
        # print("finished deleting mfcDC")
        win32gui.ReleaseDC(self.hwnd, self.hwndDC)
        # print("Released DC")
        return np.array(im)


if __name__ == "__main__":
    cap = ImageCapture("Binding of Isaac: Repentance")
    im = cap.capture_frame()
    print(im)
