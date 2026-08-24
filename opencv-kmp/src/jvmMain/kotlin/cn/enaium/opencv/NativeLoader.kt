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
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

/**
 * Resolves and loads the JNI shared library bundled in the matching
 * `opencv-kmp-jni-jvm-{os}-{arch}` artifact's resources.
 *
 * Falls back to `System.loadLibrary("opencv_jni")` if no bundled binary
 * matches the host (a developer workflow places the lib on
 * `java.library.path`).
 */
internal object NativeLoader {
    private const val LIB_NAME = "opencv_jni"
    private const val RESOURCE_BASE = "/cn/enaium/opencv/native"

    fun load() {
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
                        "classifier=$classifier.",
            )
        val bytes = stream.use { it.readBytes() }
        val target = extractToTemp(bytes, libFile)
        try {
            System.load(target.absolutePath)
        } catch (e: UnsatisfiedLinkError) {
            throw UnsatisfiedLinkError(
                "Failed to load $libFile from $resourcePath (os=${System.getProperty("os.name")}, " +
                        "arch=${System.getProperty("os.arch")}): ${e.message}",
            )
        }
    }

    private fun extractToTemp(bytes: ByteArray, libFile: String): File {
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
        val hex = digest.joinToString("") { "%02x".format(it) }.take(16)
        val tempDir = File(System.getProperty("java.io.tmpdir"), "opencv-kmp-$hex")
        tempDir.mkdirs()
        val target = File(tempDir, libFile)
        if (!target.isFile || target.length().toInt() != bytes.size) {
            val tmp = File.createTempFile(libFile, ".part", tempDir)
            tmp.writeBytes(bytes)
            Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
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
        // Android reports os.name as "Linux"; Dalvik/ART always ships
        // android.os.Build, so probe for it reflectively.
        if (runCatching { Class.forName("android.os.Build") }.isSuccess) return "android"
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
