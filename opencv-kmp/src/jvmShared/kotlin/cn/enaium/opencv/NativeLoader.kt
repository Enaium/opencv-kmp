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

import java.io.File
import java.io.InputStream

/**
 * Resolves and loads the JNI shared library for the running platform.
 *
 * Android: the per-ABI artifact is an AAR whose `jni/<abi>/libopencv_jni.so`
 * entry is packaged into the APK's `lib/<abi>/`, so the platform loader
 * (`System.loadLibrary`) finds it. The bundled resource is only consulted if
 * that fails (a consumer that strips the AAR native libs).
 *
 * Desktop: the matching `opencv-kmp-jni-jvm-{os}-{arch}` artifact carries the
 * library as a classpath resource which is extracted to a temp file and
 * `System.load`ed; `System.loadLibrary` is the fallback for a `java.library.path`
 * developer setup.
 *
 * Loading happens once per process: the three JNI holder objects call [load]
 * from their init blocks and later calls are no-ops.
 */
internal object NativeLoader {
    private const val LIB_NAME = "opencv_jni"
    private const val RESOURCE_BASE = "/cn/enaium/opencv/native"

    @Volatile
    private var loaded = false

    @Synchronized
    fun load() {
        if (loaded) return
        val android = detectOs() == "android"
        if (android && loadFromLibraryPath()) {
            loaded = true
            return
        }
        try {
            loadFromResource()
        } catch (e: UnsatisfiedLinkError) {
            if (!android && loadFromLibraryPath()) {
                loaded = true
                return
            }
            throw e
        }
        loaded = true
    }

    /** True when the platform loader resolved `libopencv_jni` (APK `lib/<abi>/`, `java.library.path`). */
    private fun loadFromLibraryPath(): Boolean = try {
        System.loadLibrary(LIB_NAME)
        true
    } catch (_: UnsatisfiedLinkError) {
        false
    }

    private fun loadFromResource() {
        val classifier = detectClassifier()
        // Windows DLLs don't use the "lib" prefix (opencv_jni.dll), while
        // Unix shared libraries do (libopencv_jni.so/.dylib).
        val prefix = if (classifier.startsWith("windows")) "" else "lib"
        val libFile = "$prefix$LIB_NAME.${libExtension()}"
        val resourcePath = "$RESOURCE_BASE/$classifier/$libFile"
        val stream = NativeLoader::class.java.getResourceAsStream(resourcePath)
            ?: throw UnsatisfiedLinkError(
                "Resource $resourcePath not found. " +
                        "os=${System.getProperty("os.name")}, arch=${System.getProperty("os.arch")}, " +
                        "classifier=$classifier. " +
                        "If this artifact is consumed on Android, the native library is packaged " +
                        "in the APK's lib/<abi>/ instead and must be loadable by the platform loader.",
            )
        val target = stream.use { extractToTemp(it, libFile) }
        try {
            System.load(target.absolutePath)
        } catch (e: UnsatisfiedLinkError) {
            throw UnsatisfiedLinkError(
                "Failed to load $libFile from $resourcePath (os=${System.getProperty("os.name")}, " +
                        "arch=${System.getProperty("os.arch")}): ${e.message}",
            )
        }
    }

    /**
     * Streams the resource into a private temp file instead of buffering it in
     * a byte array: the unstripped libraries are hundreds of megabytes on
     * Android, where the resource is a fallback only. The copy is deleted when
     * the JVM exits.
     */
    private fun extractToTemp(stream: InputStream, libFile: String): File {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "opencv-kmp")
        tempDir.mkdirs()
        val target = File.createTempFile(libFile, null, tempDir)
        target.deleteOnExit()
        target.outputStream().use { out -> stream.copyTo(out) }
        return target
    }

    private fun detectClassifier(): String {
        val os = detectOs()
        val arch = when (val osArch = System.getProperty("os.arch").orEmpty().lowercase()) {
            "amd64", "x86_64", "x64" -> "x86_64"
            "aarch64", "arm64" -> if (os == "android") "arm64-v8a" else "aarch64"
            "arm", "armv7", "armv7l", "armv8l" -> "armeabi-v7a"
            "x86", "i386", "i486", "i586", "i686" -> if (os == "android") "x86" else error(
                "Unsupported CPU architecture for opencv-kmp JVM artifact: $osArch",
            )
            else -> error("Unsupported CPU architecture for opencv-kmp JVM artifact: $osArch")
        }
        return "$os-$arch"
    }

    private fun detectOs(): String {
        // Android reports os.name as "Linux"; Dalvik/ART distinguishes itself
        // via java.vendor / java.runtime.name system properties, so detect it
        // without loading any class reflectively.
        val isAndroid = listOf("java.vendor", "java.runtime.name", "java.vm.name")
            .map { System.getProperty(it).orEmpty() }
            .any { it.contains("Android") || it.contains("Dalvik") }
        if (isAndroid) return "android"
        val osName = System.getProperty("os.name").orEmpty().lowercase()
        return when {
            osName.contains("linux") -> "linux"
            osName.contains("mac") || osName.contains("darwin") || osName.contains("osx") -> "darwin"
            osName.contains("win") -> "windows"
            else -> error("Unsupported OS for opencv-kmp JVM artifact: $osName")
        }
    }

    private fun libExtension(): String {
        val osName = System.getProperty("os.name").orEmpty().lowercase()
        return when {
            osName.contains("win") -> "dll"
            osName.contains("mac") || osName.contains("darwin") || osName.contains("osx") -> "dylib"
            else -> "so"
        }
    }
}
