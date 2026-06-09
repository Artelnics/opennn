!include x64.nsh
Name "libjpeg-turbo SDK for Visual C++ 64-bit"
OutFile "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}libjpeg-turbo-3.1.4-vc-x64.exe"
InstallDir "C:\Users\Roberto\Documents\opennn\build-ninja\_deps\opennn_libjpeg_turbo-install"

SetCompressor bzip2

Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

Section "libjpeg-turbo SDK for Visual C++ 64-bit (required)"
!ifdef WIN64
	${If} ${RunningX64}
	${DisableX64FSRedirection}
	${Endif}
!endif
	SectionIn RO
!ifdef GCC
	IfFileExists $SYSDIR/libturbojpeg.dll exists 0
!else
	IfFileExists $SYSDIR/turbojpeg.dll exists 0
!endif
	goto notexists
	exists:
!ifdef GCC
	MessageBox MB_OK "An existing version of the libjpeg-turbo SDK for Visual C++ 64-bit is already installed.  Please uninstall it first."
!else
	MessageBox MB_OK "An existing version of the libjpeg-turbo SDK for Visual C++ 64-bit or the TurboJPEG SDK is already installed.  Please uninstall it first."
!endif
	quit

	notexists:
	SetOutPath $SYSDIR
!ifdef GCC
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\libturbojpeg.dll"
!else
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}turbojpeg.dll"
!endif
	SetOutPath $INSTDIR\bin
!ifdef GCC
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\libturbojpeg.dll"
!else
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}turbojpeg.dll"
!endif
!ifdef GCC
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\libjpeg-62.dll"
!else
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}jpeg62.dll"
!endif
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}cjpeg.exe"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}djpeg.exe"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}jpegtran.exe"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}tjbench.exe"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}rdjpgcom.exe"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}wrjpgcom.exe"
	SetOutPath $INSTDIR\lib
!ifdef GCC
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\libturbojpeg.dll.a"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\libturbojpeg.a"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\libjpeg.dll.a"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\libjpeg.a"
!else
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}turbojpeg.lib"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}turbojpeg-static.lib"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}jpeg.lib"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\${BUILDDIR}jpeg-static.lib"
!endif
	SetOutPath $INSTDIR\lib\pkgconfig
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\pkgscripts\libjpeg.pc"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\pkgscripts\libturbojpeg.pc"
	SetOutPath $INSTDIR\lib\cmake\libjpeg-turbo
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\pkgscripts\libjpeg-turboConfig.cmake"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\pkgscripts\libjpeg-turboConfigVersion.cmake"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\win\libjpeg-turboTargets.cmake"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\win\libjpeg-turboTargets-release.cmake"
!ifdef JAVA
	SetOutPath $INSTDIR\classes
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\java\turbojpeg.jar"
!endif
	SetOutPath $INSTDIR\include
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep-build\jconfig.h"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\src\jerror.h"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\src\jmorecfg.h"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\src\jpeglib.h"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\src\turbojpeg.h"
	SetOutPath $INSTDIR\doc
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\README.ijg"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\README.md"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\LICENSE.md"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\src\example.c"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\doc\libjpeg.txt"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\doc\structure.txt"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\doc\usage.txt"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\doc\wizard.txt"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\src\tjcomp.c"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\src\tjdecomp.c"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\src\tjtran.c"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\java\TJComp.java"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\java\TJDecomp.java"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\java\TJTran.java"
!ifdef GCC
	SetOutPath $INSTDIR\man\man1
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\doc\cjpeg.1"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\doc\djpeg.1"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\doc\jpegtran.1"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\doc\rdjpgcom.1"
	File "C:/Users/Roberto/Documents/opennn/build-ninja/opennn/opennn_libjpeg_turbo_ep-prefix/src/opennn_libjpeg_turbo_ep\doc\wrjpgcom.1"
!endif

	WriteRegStr HKLM "SOFTWARE\libjpeg-turbo64 3.1.4" "Install_Dir" "$INSTDIR"

	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\libjpeg-turbo64 3.1.4" "DisplayName" "libjpeg-turbo SDK v3.1.4 for Visual C++ 64-bit"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\libjpeg-turbo64 3.1.4" "UninstallString" '"$INSTDIR\uninstall_3.1.4.exe"'
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\libjpeg-turbo64 3.1.4" "NoModify" 1
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\libjpeg-turbo64 3.1.4" "NoRepair" 1
	WriteUninstaller "uninstall_3.1.4.exe"
SectionEnd

Section "Uninstall"
!ifdef WIN64
	${If} ${RunningX64}
	${DisableX64FSRedirection}
	${Endif}
!endif

	SetShellVarContext all

	DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\libjpeg-turbo64 3.1.4"
	DeleteRegKey HKLM "SOFTWARE\libjpeg-turbo64 3.1.4"

!ifdef GCC
	Delete $INSTDIR\bin\libjpeg-62.dll
	Delete $INSTDIR\bin\libturbojpeg.dll
	Delete $SYSDIR\libturbojpeg.dll
	Delete $INSTDIR\lib\libturbojpeg.dll.a
	Delete $INSTDIR\lib\libturbojpeg.a
	Delete $INSTDIR\lib\libjpeg.dll.a
	Delete $INSTDIR\lib\libjpeg.a
!else
	Delete $INSTDIR\bin\jpeg62.dll
	Delete $INSTDIR\bin\turbojpeg.dll
	Delete $SYSDIR\turbojpeg.dll
	Delete $INSTDIR\lib\jpeg.lib
	Delete $INSTDIR\lib\jpeg-static.lib
	Delete $INSTDIR\lib\turbojpeg.lib
	Delete $INSTDIR\lib\turbojpeg-static.lib
!endif
	Delete $INSTDIR\lib\pkgconfig\libjpeg.pc
	Delete $INSTDIR\lib\pkgconfig\libturbojpeg.pc
	Delete $INSTDIR\lib\cmake\libjpeg-turbo\libjpeg-turboConfig.cmake
	Delete $INSTDIR\lib\cmake\libjpeg-turbo\libjpeg-turboConfigVersion.cmake
	Delete $INSTDIR\lib\cmake\libjpeg-turbo\libjpeg-turboTargets.cmake
	Delete $INSTDIR\lib\cmake\libjpeg-turbo\libjpeg-turboTargets-release.cmake
!ifdef JAVA
	Delete $INSTDIR\classes\turbojpeg.jar
!endif
	Delete $INSTDIR\bin\cjpeg.exe
	Delete $INSTDIR\bin\djpeg.exe
	Delete $INSTDIR\bin\jpegtran.exe
	Delete $INSTDIR\bin\tjbench.exe
	Delete $INSTDIR\bin\rdjpgcom.exe
	Delete $INSTDIR\bin\wrjpgcom.exe
	Delete $INSTDIR\include\jconfig.h
	Delete $INSTDIR\include\jerror.h
	Delete $INSTDIR\include\jmorecfg.h
	Delete $INSTDIR\include\jpeglib.h
	Delete $INSTDIR\include\turbojpeg.h
	Delete $INSTDIR\uninstall_3.1.4.exe
	Delete $INSTDIR\doc\README.ijg
	Delete $INSTDIR\doc\README.md
	Delete $INSTDIR\doc\LICENSE.md
	Delete $INSTDIR\doc\example.c
	Delete $INSTDIR\doc\libjpeg.txt
	Delete $INSTDIR\doc\structure.txt
	Delete $INSTDIR\doc\usage.txt
	Delete $INSTDIR\doc\wizard.txt
	Delete $INSTDIR\doc\tjcomp.c
	Delete $INSTDIR\doc\tjdecomp.c
	Delete $INSTDIR\doc\tjtran.c
	Delete $INSTDIR\doc\TJComp.java
	Delete $INSTDIR\doc\TJDecomp.java
	Delete $INSTDIR\doc\TJTran.java
!ifdef GCC
	Delete $INSTDIR\man\man1\cjpeg.1
	Delete $INSTDIR\man\man1\djpeg.1
	Delete $INSTDIR\man\man1\jpegtran.1
	Delete $INSTDIR\man\man1\rdjpgcom.1
	Delete $INSTDIR\man\man1\wrjpgcom.1
!endif

	RMDir "$INSTDIR\include"
	RMDir "$INSTDIR\lib\pkgconfig"
	RMDir "$INSTDIR\lib\cmake\libjpeg-turbo"
	RMDir "$INSTDIR\lib\cmake"
	RMDir "$INSTDIR\lib"
	RMDir "$INSTDIR\doc"
!ifdef GCC
	RMDir "$INSTDIR\man\man1"
	RMDir "$INSTDIR\man"
!endif
!ifdef JAVA
	RMDir "$INSTDIR\classes"
!endif
	RMDir "$INSTDIR\bin"
	RMDir "$INSTDIR"

SectionEnd
