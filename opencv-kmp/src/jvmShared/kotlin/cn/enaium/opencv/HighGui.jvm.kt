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
package cn.enaium.opencv

/**
 * JVM/Android actuals backed by libopencv_jni.
 *
 * Windows (Win32UI) and macOS (Cocoa) show real windows; Linux artifacts are
 * built headless so every call reports on stderr instead of opening one;
 * Android has no desktop GUI at all.
 */
actual fun namedWindow(winName: String, flags: Int) = Jni.namedWindow(winName, flags)

actual fun resizeWindow(winName: String, width: Int, height: Int) =
    Jni.resizeWindow(winName, width, height)

actual fun imshow(winName: String, mat: Mat) = Jni.imshow(winName, handleOf(mat))

actual fun waitKey(delayMs: Int): Int = Jni.waitKey(delayMs)

actual fun destroyWindow(winName: String) = Jni.destroyWindow(winName)

actual fun destroyAllWindows() = Jni.destroyAllWindows()