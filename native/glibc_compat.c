/*
 * Copyright (c) 2026 Enaium
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
 * glibc compatibility shims, Linux only.
 *
 * OpenCV compiled on a modern distro (glibc >= 2.38) references symbols that
 * the Kotlin/Native bundled Linux sysroot (glibc 2.19) does not provide:
 *
 *  - __isoc23_strtol/strtoul/strtoll/strtoull/strtof/strtod/strtold and
 *    __isoc23_sscanf/vsscanf/fscanf/vfscanf: glibc 2.38+ compiles calls to
 *    the C23-semantics variants when _GNU_SOURCE is defined. That redirect
 *    is an asm-name-level compiler attribute, NOT a macro, so these wrappers
 *    reach the real libc symbols through explicit asm labels (calling strtol
 *    directly from this file would be renamed straight back into
 *    __isoc23_strtol and recurse forever).
 *  - __libc_single_threaded: glibc 2.32+ data symbol referenced by modern
 *    C/C++ runtime bits.
 *
 * Every definition lives in its own archive member; the linker only pulls
 * them when a reference exists, so they are harmless on systems where the
 * real symbols are available.
 */
#if defined(__linux__) && !defined(__ANDROID__)

#define _GNU_SOURCE 1
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

extern long int cvk_glibc_strtol(const char *nptr, char **endptr, int base) __asm__("strtol");
extern unsigned long cvk_glibc_strtoul(const char *nptr, char **endptr, int base) __asm__("strtoul");
extern long long cvk_glibc_strtoll(const char *nptr, char **endptr, int base) __asm__("strtoll");
extern unsigned long long cvk_glibc_strtoull(const char *nptr, char **endptr, int base) __asm__("strtoull");
extern float cvk_glibc_strtof(const char *nptr, char **endptr) __asm__("strtof");
extern double cvk_glibc_strtod(const char *nptr, char **endptr) __asm__("strtod");
extern long double cvk_glibc_strtold(const char *nptr, char **endptr) __asm__("strtold");
extern int cvk_glibc_vfscanf(void *stream, const char *format, va_list ap) __asm__("vfscanf");
extern int cvk_glibc_vsscanf(const char *s, const char *format, va_list ap) __asm__("vsscanf");

long __isoc23_strtol(const char *nptr, char **endptr, int base)
{
    return cvk_glibc_strtol(nptr, endptr, base);
}

unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base)
{
    return cvk_glibc_strtoul(nptr, endptr, base);
}

long long __isoc23_strtoll(const char *nptr, char **endptr, int base)
{
    return cvk_glibc_strtoll(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull(const char *nptr, char **endptr, int base)
{
    return cvk_glibc_strtoull(nptr, endptr, base);
}

float __isoc23_strtof(const char *nptr, char **endptr)
{
    return cvk_glibc_strtof(nptr, endptr);
}

double __isoc23_strtod(const char *nptr, char **endptr)
{
    return cvk_glibc_strtod(nptr, endptr);
}

long double __isoc23_strtold(const char *nptr, char **endptr)
{
    return cvk_glibc_strtold(nptr, endptr);
}

int __isoc23_vsscanf(const char *s, const char *format, va_list ap)
{
    return cvk_glibc_vsscanf(s, format, ap);
}

int __isoc23_vfscanf(void *stream, const char *format, va_list ap)
{
    return cvk_glibc_vfscanf(stream, format, ap);
}

int __isoc23_sscanf(const char *s, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    int result = cvk_glibc_vsscanf(s, format, ap);
    va_end(ap);
    return result;
}

int __isoc23_fscanf(void *stream, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    int result = cvk_glibc_vfscanf(stream, format, ap);
    va_end(ap);
    return result;
}

int __libc_single_threaded = 1;

#endif /* __linux__ && !__ANDROID__ */
