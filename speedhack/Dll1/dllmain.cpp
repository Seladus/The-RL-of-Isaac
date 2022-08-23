// dllmain.cpp : Définit le point d'entrée de l'application DLL.
#include "pch.h"
#include <Windows.h>
#include <iostream>
#include <filesystem>
#include "detours.h"

namespace fs = std::filesystem;

static BOOL(WINAPI* OriginalQueryPerformanceCounter)(LARGE_INTEGER* performanceCounter) = QueryPerformanceCounter;

DWORD speedMultiplier = 500;
int64_t BasePerformanceCount;

BOOL WINAPI NewQueryPerformanceCounter(LARGE_INTEGER* performanceCounter) {
    int64_t currentPerformanceCount;
    if (!OriginalQueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&currentPerformanceCount))) return FALSE;
    auto newTime = currentPerformanceCount + ((currentPerformanceCount - BasePerformanceCount) * speedMultiplier);
    *performanceCounter = *reinterpret_cast<LARGE_INTEGER*>(&newTime);
    return TRUE;
}

void StartConsole() {
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);
    std::cout << "OKOK LEZGO" << std::endl;
}

DWORD WINAPI onAttach(HMODULE hModule) {
    //StartConsole();
    DetourRestoreAfterWith();
    std::cout << "Detouring at address " << OriginalQueryPerformanceCounter << " with " << NewQueryPerformanceCounter << std::endl;
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourAttach(&(PVOID&)OriginalQueryPerformanceCounter, NewQueryPerformanceCounter);
    DetourTransactionCommit();
    return 0;
}


BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        //CloseHandle(CreateThread(nullptr, 0, (LPTHREAD_START_ROUTINE)onAttach, hModule, 0, nullptr));;
        onAttach(nullptr);
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

