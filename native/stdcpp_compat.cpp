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
 * libstdc++ compatibility shims, Linux only.
 *
 * OpenCV objects compiled by a recent host GCC expect explicitly-instanced
 * std::__cxx11 stream entry points (e.g. basic_stringstream<char>::
 * basic_stringstream()) that the old libstdc++.so shipped inside the
 * Kotlin/Native sysroot (GCC 8.3) does not export. Compiling an explicit
 * instantiation definition here makes the host toolchain emit strong local
 * definitions; the linker then never needs them from the ancient copy.
 */
#include <new>
#include <sstream>

// Targets whose klib embeds the full modern libstdc++ archive (see the
// CVK_SKIP_STDCPP_SHIM define in CMakeLists.txt) must not duplicate the
// helper symbols the real archive already provides.
#ifndef CVK_SKIP_STDCPP_SHIM

// GCC 12+ inline helper; sysroots whose libstdc++ predates it (Kotlin/Native
// Linux and MinGW) need a strong definition. The Android NDK ships its own,
// so this TU must stay silent there or the symbol gets redefined.
#if (defined(__linux__) && !defined(__ANDROID__)) || defined(_WIN32)
namespace std {
[[noreturn]] void __throw_bad_array_new_length() { throw bad_array_new_length(); }
} // namespace std
#endif

#if defined(__linux__) && !defined(__ANDROID__)

template class std::basic_stringstream<char, std::char_traits<char>, std::allocator<char>>;

#endif /* __linux__ && !__ANDROID__ */

#endif /* CVK_SKIP_STDCPP_SHIM */
