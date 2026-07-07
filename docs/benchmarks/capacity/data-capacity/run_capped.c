/* Run a child process inside a Windows Job Object with a hard committed-memory
 * cap, so an out-of-memory condition happens at a fixed, reproducible budget
 * instead of dragging the whole machine into pagefile thrashing.
 *
 *   usage:  run_capped <cap_bytes> <program> [args...]
 *
 * Prints the child's exit code as "child_exit=<n>" and "capped=1". A child that
 * exceeds the cap fails its allocation (our test programs catch this and print
 * RESULT=OOM) or is terminated by the job; either way the cap is the ceiling.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

int wmain(int argc, wchar_t** argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: run_capped <cap_bytes> <program> [args...]\n");
        return 2;
    }

    const unsigned long long cap = _wcstoui64(argv[1], NULL, 10);

    HANDLE job = CreateJobObjectW(NULL, NULL);
    if (!job) { fprintf(stderr, "CreateJobObject failed\n"); return 1; }

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits;
    memset(&limits, 0, sizeof(limits));
    limits.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    limits.JobMemoryLimit = (SIZE_T)cap;

    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                 &limits, sizeof(limits)))
    {
        fprintf(stderr, "SetInformationJobObject failed (%lu)\n", GetLastError());
        return 1;
    }

    /* Rebuild the child command line from argv[2..]. */
    wchar_t cmdline[32768];
    cmdline[0] = L'\0';
    for (int i = 2; i < argc; i++)
    {
        if (i > 2) wcscat_s(cmdline, 32768, L" ");
        /* quote args that contain spaces */
        if (wcschr(argv[i], L' '))
        {
            wcscat_s(cmdline, 32768, L"\"");
            wcscat_s(cmdline, 32768, argv[i]);
            wcscat_s(cmdline, 32768, L"\"");
        }
        else
            wcscat_s(cmdline, 32768, argv[i]);
    }

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    memset(&pi, 0, sizeof(pi));

    /* CREATE_SUSPENDED so we can assign to the job before it runs. */
    if (!CreateProcessW(NULL, cmdline, NULL, NULL, TRUE,
                        CREATE_SUSPENDED, NULL, NULL, &si, &pi))
    {
        fprintf(stderr, "CreateProcess failed (%lu)\n", GetLastError());
        return 1;
    }

    if (!AssignProcessToJobObject(job, pi.hProcess))
    {
        fprintf(stderr, "AssignProcessToJobObject failed (%lu)\n", GetLastError());
        TerminateProcess(pi.hProcess, 1);
        return 1;
    }

    ResumeThread(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);

    printf("capped=1\n");
    printf("child_exit=%lu\n", code);

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(job);   /* KILL_ON_JOB_CLOSE cleans up any stragglers */
    return (int)code;
}
