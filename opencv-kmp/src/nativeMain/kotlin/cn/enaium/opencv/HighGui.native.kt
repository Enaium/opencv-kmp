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
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package cn.enaium.opencv

import cvk.cvk_destroy_all_windows
import cvk.cvk_destroy_window
import cvk.cvk_imshow
import cvk.cvk_named_window
import cvk.cvk_resize_window
import cvk.cvk_wait_key

/**
 * Kotlin/Native actuals backed by the cvk shim. Window backends: Win32UI on
 * Windows and Cocoa on macOS; Linux artifacts are built headless (calls log
 * to stderr), and Android has no desktop GUI at all.
 */
actual fun namedWindow(winName: String, flags: Int) = cvk_named_window(winName, flags)

actual fun resizeWindow(winName: String, width: Int, height: Int) =
    cvk_resize_window(winName, width, height)

actual fun imshow(winName: String, mat: Mat) = cvk_imshow(winName, mat.nativeHandle())

actual fun waitKey(delayMs: Int): Int = cvk_wait_key(delayMs)

actual fun destroyWindow(winName: String) = cvk_destroy_window(winName)

actual fun destroyAllWindows() = cvk_destroy_all_windows()