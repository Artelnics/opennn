%global _docdir %{_defaultdocdir}/%{name}-%{version}
%define _prefix  /home/artelnics/Documents/openNN_refactor/opennn/build-cpu/_deps/opennn_libjpeg_turbo-install
%define _bindir  /home/artelnics/Documents/openNN_refactor/opennn/build-cpu/_deps/opennn_libjpeg_turbo-install/bin
%define _datarootdir  /home/artelnics/Documents/openNN_refactor/opennn/build-cpu/_deps/opennn_libjpeg_turbo-install/share
%define _includedir  /home/artelnics/Documents/openNN_refactor/opennn/build-cpu/_deps/opennn_libjpeg_turbo-install/include
%define _mandir  /home/artelnics/Documents/openNN_refactor/opennn/build-cpu/_deps/opennn_libjpeg_turbo-install/share/man
%define _enable_static  1
%define _enable_shared  0
%define _with_turbojpeg  0
%define _with_java  0
%if "%{_with_java}" == "1"
%define _javadir  
%endif

%if "%{?__isa_bits:1}" == "1"
%define _bits  %{__isa_bits}
%else
# RPM < 4.6
%if "%{_lib}" == "lib64"
%define _bits  64
%else
%define _bits  32
%endif
%endif

%if "%{_bits}" == "64"
%define _syslibdir /usr/lib64
%else
%define _syslibdir /usr/lib
%endif

#-->%if 1
%if "%{_bits}" == "64"
%define _libdir  %{_exec_prefix}/lib64
%else
%if "%{_prefix}" == "/opt/libjpeg-turbo"
%define _libdir  %{_exec_prefix}/lib32
%endif
%endif
#-->%else
%define _libdir  /home/artelnics/Documents/openNN_refactor/opennn/build-cpu/_deps/opennn_libjpeg_turbo-install/lib
#-->%endif

%define __requires_exclude_from  ^%{_libdir}/cmake/libjpeg-turbo/.*$

Summary: A SIMD-accelerated JPEG codec that provides both the libjpeg and TurboJPEG APIs
Name: libjpeg-turbo
Version: 3.1.4
Vendor: The libjpeg-turbo Project
URL: https://libjpeg-turbo.org
Group: System Environment/Libraries
#-->Source0: https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/%{version}/libjpeg-turbo-%{version}.tar.gz
Release: 20260617
License: BSD-style
BuildRoot: %{_blddir}/%{name}-buildroot-%{version}-%{release}
Requires: /sbin/ldconfig
%if "%{_bits}" == "64"
Provides: %{name} = %{version}-%{release}, libjpeg-turbo = %{version}-%{release}, libturbojpeg.so()(64bit)
%else
Provides: %{name} = %{version}-%{release}, libjpeg-turbo = %{version}-%{release}, libturbojpeg.so
%endif

%description
libjpeg-turbo is a JPEG image codec that uses SIMD instructions to accelerate
baseline JPEG compression and decompression on x86, x86-64, Arm, PowerPC, and
MIPS systems, as well as progressive JPEG compression on x86, x86-64, and Arm
systems.  On such systems, libjpeg-turbo is generally 2-6x as fast as libjpeg,
all else being equal.  On other types of systems, libjpeg-turbo can still
outperform libjpeg by a significant amount, by virtue of its highly-optimized
Huffman coding routines.  In many cases, the performance of libjpeg-turbo
rivals that of proprietary high-speed JPEG codecs.

libjpeg-turbo implements both the traditional libjpeg API as well as the less
powerful but more straightforward TurboJPEG API.  libjpeg-turbo also features
colorspace extensions that allow it to compress from/decompress to 32-bit and
big-endian pixel buffers (RGBX, XBGR, etc.), as well as a full-featured Java
interface.

libjpeg-turbo was originally based on libjpeg/SIMD, an MMX-accelerated
derivative of libjpeg v6b developed by Miyasaka Masaru.  The TigerVNC and
VirtualGL projects made numerous enhancements to the codec in 2009, and in
early 2010, libjpeg-turbo spun off into an independent project, with the goal
of making high-speed JPEG compression/decompression technology available to a
broader range of users and developers.  libjpeg-turbo is an ISO/IEC and ITU-T
reference implementation of the JPEG standard.

#-->%prep
#-->%setup -q -n libjpeg-turbo-%{version}

#-->%build
#-->cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
#-->  -DBUILD=%{release} \
#-->  -DCMAKE_INSTALL_BINDIR=%{_bindir} \
#-->  -DCMAKE_INSTALL_DATAROOTDIR=%{_datarootdir} \
#-->  -DCMAKE_INSTALL_DOCDIR=%{_docdir} \
#-->  -DCMAKE_INSTALL_INCLUDEDIR=%{_includedir} \
#-->  -DCMAKE_INSTALL_JAVADIR=%{_javadir} \
#-->  -DCMAKE_INSTALL_LIBDIR=%{_libdir} \
#-->  -DCMAKE_INSTALL_MANDIR=%{_mandir} \
#-->  -DCMAKE_INSTALL_PREFIX=%{_prefix} \
#-->  -DCMAKE_POSITION_INDEPENDENT_CODE=0 \
#-->  -DENABLE_SHARED=0 -DENABLE_STATIC=1 \
#-->  -DSO_MAJOR_VERSION=62 \
#-->  -DSO_MINOR_VERSION=0 \
#-->  -DJPEG_LIB_VERSION=62 \
#-->  -DREQUIRE_SIMD=0 \
#-->  -DWITH_ARITH_DEC=1 -DWITH_ARITH_ENC=1 \
#-->  -DWITH_JAVA=0 \
#-->  -DWITH_JPEG7=0 -DWITH_JPEG8=0 \
#-->  -DWITH_SIMD=0 -DWITH_TURBOJPEG=0 .
#-->make DESTDIR=$RPM_BUILD_ROOT

%install
rm -rf $RPM_BUILD_ROOT
DESTDIR=$RPM_BUILD_ROOT /usr/bin/gmake install
/sbin/ldconfig -n $RPM_BUILD_ROOT%{_libdir}

#-->%if 0

# This is only needed to support in-tree RPM generation via 'make rpm'.  When
# building from a SRPM, we control where things are installed via CMake
# variables.

safedirmove ()
{
	if [ "$1" = "$2" ]; then
		return 0
	fi
	if [ "$1" = "" -o ! -d "$1" ]; then
		echo safedirmove: source dir $1 is not valid
		return 1
	fi
	if [ "$2" = "" -o -e "$2" ]; then
		echo safedirmove: dest dir $2 is not valid
		return 1
	fi
	if [ "$3" = "" -o -e "$3" ]; then
		echo safedirmove: tmp dir $3 is not valid
		return 1
	fi
	mkdir -p $3
	mv $1/* $3/
	rmdir $1
	mkdir -p $2
	mv $3/* $2/
	rmdir $3
	return 0
}

LJT_DOCDIR=/home/artelnics/Documents/openNN_refactor/opennn/build-cpu/_deps/opennn_libjpeg_turbo-install/share/doc/libjpeg-turbo
if [ ! "$LJT_DOCDIR" = "%{_docdir}" ]; then
	safedirmove $RPM_BUILD_ROOT/$LJT_DOCDIR $RPM_BUILD_ROOT/%{_docdir} $RPM_BUILD_ROOT/__tmpdoc
fi

#-->%endif

LJT_DOCDIR=/home/artelnics/Documents/openNN_refactor/opennn/build-cpu/_deps/opennn_libjpeg_turbo-install/share/doc/libjpeg-turbo
if [ "%{_prefix}" = "/opt/libjpeg-turbo" -a "$LJT_DOCDIR" = "/opt/libjpeg-turbo/doc" ]; then
	ln -fs %{_docdir} $RPM_BUILD_ROOT/$LJT_DOCDIR
fi

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root)
%dir %{_docdir}
%doc %{_docdir}/*
%dir %{_prefix}
%if "%{_prefix}" == "/opt/libjpeg-turbo" && "%{_docdir}" != "%{_prefix}/doc"
	%{_prefix}/doc
%endif
%dir %{_bindir}
%{_bindir}/cjpeg
%{_bindir}/djpeg
%{_bindir}/jpegtran
%if "%{_with_turbojpeg}" == "1"
	%{_bindir}/tjbench
%endif
%{_bindir}/rdjpgcom
%{_bindir}/wrjpgcom
%if "%{_libdir}" != "%{_syslibdir}"
%dir %{_libdir}
%endif
%if "%{_enable_shared}" == "1"
	%{_libdir}/libjpeg.so.62.4.0
	%{_libdir}/libjpeg.so.62
	%{_libdir}/libjpeg.so
%endif
%if "%{_enable_static}" == "1"
	%{_libdir}/libjpeg.a
%endif
%dir %{_libdir}/pkgconfig
%{_libdir}/pkgconfig/libjpeg.pc
%dir %{_libdir}/cmake
%dir %{_libdir}/cmake/libjpeg-turbo
%{_libdir}/cmake/libjpeg-turbo/*
%if "%{_with_turbojpeg}" == "1"
	%if "%{_enable_shared}" == "1" || "%{_with_java}" == "1"
		%{_libdir}/libturbojpeg.so.0.4.0
		%{_libdir}/libturbojpeg.so.0
		%{_libdir}/libturbojpeg.so
	%endif
	%if "%{_enable_static}" == "1"
		%{_libdir}/libturbojpeg.a
	%endif
	%{_libdir}/pkgconfig/libturbojpeg.pc
%endif
%dir %{_includedir}
%{_includedir}/jconfig.h
%{_includedir}/jerror.h
%{_includedir}/jmorecfg.h
%{_includedir}/jpeglib.h
%if "%{_with_turbojpeg}" == "1"
	%{_includedir}/turbojpeg.h
%endif
%dir %{_mandir}
%dir %{_mandir}/man1
%{_mandir}/man1/cjpeg.1*
%{_mandir}/man1/djpeg.1*
%{_mandir}/man1/jpegtran.1*
%{_mandir}/man1/rdjpgcom.1*
%{_mandir}/man1/wrjpgcom.1*
%if "%{_prefix}" != "%{_datarootdir}"
	%dir %{_datarootdir}
%endif
%if "%{_with_java}" == "1"
	%dir %{_javadir}
	%{_javadir}/turbojpeg.jar
%endif

%changelog
