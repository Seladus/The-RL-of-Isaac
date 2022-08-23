// Dll Injector.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <iostream>
#include <Windows.h>
#include <TlHelp32.h>
#include "fileapi.h"
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;


DWORD GetProcId(const wchar_t* procName) {
    DWORD procId = 0;
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);

    if (hSnap != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32 procEntry;
        procEntry.dwSize = sizeof(procEntry);

        if (Process32First(hSnap, &procEntry)) {
            do {
                if (!wcscmp(procEntry.szExeFile, procName)) {
                    procId = procEntry.th32ProcessID;
                    break;
                }
            } while (Process32Next(hSnap, &procEntry));
        }
    }
    CloseHandle(hSnap);
    return procId;
}


int wmain(int argc, wchar_t** argv)
{
    fs::path p;
    if (argc > 1) {
        p = argv[1];
    }
    else {
        cout << "Please specifiy which Dll to inject !" << endl;
        return 1;
    }

    wchar_t* windowName = nullptr;
    if (argc > 2) {
        windowName = argv[2];
    } else {
        cout << "Please specify which window to hook !" << endl;
        return 1;
    }
    string absolutePath = fs::absolute(p).string();
    const char* dllPath = &absolutePath[0];
    
    DWORD procId = 0;

    while (!procId) {
        HWND hw = FindWindow(NULL, L"Binding of Isaac: Repentance");
        DWORD dwThreadId = GetWindowThreadProcessId(hw, &procId);
        Sleep(30);
    }

    HANDLE hProc = OpenProcess(PROCESS_ALL_ACCESS, 0, procId);

    if (hProc && hProc != INVALID_HANDLE_VALUE) {
        void* loc = VirtualAllocEx(hProc, 0, MAX_PATH, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        WriteProcessMemory(hProc, loc, dllPath, strlen(dllPath) + 1, 0);

        cout << "Creating remote thread -> Loading Dll in process..." << endl;
        HANDLE hThread = CreateRemoteThread(hProc, 0, 0, (LPTHREAD_START_ROUTINE)LoadLibraryA, loc, 0, 0);

        if (hThread) {
            cout << "Dll successfully loaded in process at address : " << hThread << endl;
            CloseHandle(hThread);
        }
    }

    if (hProc) {
        CloseHandle(hProc);
    }

    return 0;
}
