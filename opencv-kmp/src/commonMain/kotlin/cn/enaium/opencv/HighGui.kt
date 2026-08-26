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

/** Window/backend flags accepted by [namedWindow] (cv::WindowFlags). */
object WindowFlags {
    /** User can resize the window; no size constraint. */
    const val WINDOW_NORMAL = 0x00000000

    /** Window fits the displayed image and cannot be resized by hand. */
    const val WINDOW_AUTOSIZE = 0x00000001

    /** Draw with a plain GDI/SDL-less chrome (no status/tool bar). */
    const val WINDOW_GUI_NORMAL = 0x00000010

    /** Draw with full window chrome. */
    const val WINDOW_GUI_EXPANDED = 0x00000080
}

/**
 * Minimal HighGUI surface (cv::namedWindow/imshow/waitKey family).
 *
 * Real window backends exist on Windows (Win32UI) and macOS (Cocoa); Linux
 * artifacts are built headless and Android has no desktop GUI, where calls
 * either throw or report on stderr — see the platform notes on each actual.
 */
expect fun namedWindow(winName: String, flags: Int = WindowFlags.WINDOW_AUTOSIZE)

/** Resizes an existing [winName] window (`cv::resizeWindow`). */
expect fun resizeWindow(winName: String, width: Int, height: Int)

/** Shows [mat] inside [winName], creating the window on demand. */
expect fun imshow(winName: String, mat: Mat)

/**
 * Waits up to [delayMs] milliseconds for a key press (0 = infinite).
 * Returns the key code, or -1 when the timeout elapsed.
 */
expect fun waitKey(delayMs: Int = 0): Int

/** Destroys a single window (`cv::destroyWindow`). */
expect fun destroyWindow(winName: String)

/** Destroys every OpenCV window (`cv::destroyAllWindows`). */
expect fun destroyAllWindows()