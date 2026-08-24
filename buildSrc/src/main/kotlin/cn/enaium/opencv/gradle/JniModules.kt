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
package cn.enaium.opencv.gradle

import org.gradle.api.Project
import org.gradle.api.tasks.Copy
import org.gradle.api.tasks.Exec
import org.gradle.internal.os.OperatingSystem

import java.io.File

/**
 * Shared setup for the per-classifier JNI artifact modules (`:jni-jvm-*`).
 *
 * Each module calls [configure] with its classifier; this wires up the
 * CMake build of `libopencv_jni` for that OS/arch (host-native where
 * possible, cross-compiled where a toolchain exists), packages the shared
 * library as a classpath resource under
 * `/cn/enaium/opencv/native/<classifier>/`, and publishes it as
 * `cn.enaium.opencv:opencv-kmp-jni-jvm-<classifier>`.
 */
object JniModules {

    fun configure(project: Project, classifier: String) {
        val host = OperatingSystem.current()
        val hostArch = System.getProperty("os.arch").lowercase()

        fun hasToolchain(executable: String): Boolean =
            System.getenv("PATH")?.split(File.pathSeparator).orEmpty().any { dir ->
                File(dir, executable).let { it.isFile && it.canExecute() }
            }

        // Locates the highest NDK under $ANDROID_HOME / sdk.dir / ~/Android/Sdk.
        fun ndkPath(): String? {
            val home = System.getenv("ANDROID_HOME")
                ?: System.getenv("ANDROID_SDK_ROOT")
                ?: project.rootProject.file("local.properties").takeIf { it.isFile }
                    ?.readLines()
                    ?.firstOrNull { it.trimStart().startsWith("sdk.dir=") }
                    ?.substringAfter('=')?.trim()
                ?: System.getProperty("user.home") + "/Android/Sdk"
            return File(home, "ndk").listFiles()
                ?.filter { it.isDirectory && it.name.matches(Regex("\\d+\\.\\d+\\.\\d+.*")) }
                ?.maxByOrNull { it.name }
                ?.absolutePath
        }

        // NDK prebuilt toolchains live under toolchains/llvm/prebuilt/<tag>/;
        // discover the tag instead of hardcoding a host-specific directory.
        fun ndkSysrootInclude(): String? {
            val ndk = ndkPath() ?: return null
            val tag = File(ndk, "toolchains/llvm/prebuilt").listFiles()
                ?.firstOrNull { it.isDirectory } ?: return null
            return File(tag, "sysroot/usr/include").absolutePath
        }

        val parts = classifier.split('-')
        require(parts.size >= 2) { "invalid classifier '$classifier'" }

        val buildability: Pair<Boolean, List<String>> = when {
            classifier == "linux-x86_64" ->
                (host.isLinux && hostArch in setOf("amd64", "x86_64")) to listOf(
                    "-DWITH_KLEIDICV=OFF",
                    "-DWITH_CAROTENE=OFF",
                )

            classifier == "linux-aarch64" ->
                ((host.isLinux && hostArch in setOf("aarch64", "arm64")) ||
                        hasToolchain("aarch64-linux-gnu-gcc")) to listOf(
                    "-DCMAKE_SYSTEM_NAME=Linux",
                    "-DCMAKE_SYSTEM_PROCESSOR=aarch64",
                    "-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc",
                    "-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++",
                )

            classifier.startsWith("darwin-") -> {
                val arch = when (parts.last()) {
                    "aarch64" -> "arm64"
                    else -> parts.last()
                }
                (host.isMacOsX && arch in setOf("arm64", "x86_64")) to listOf(
                    "-DCMAKE_SYSTEM_NAME=Darwin",
                    "-DCMAKE_OSX_ARCHITECTURES=${if (arch == "arm64") "arm64" else "x86_64"}",
                ) + if (arch == "x86_64") listOf(
                    "-DWITH_KLEIDICV=OFF",
                    "-DWITH_CAROTENE=OFF",
                ) else emptyList()
            }

            classifier == "windows-x86_64" ->
                (host.isWindows && hasToolchain("g++.exe")) to listOf(
                    "-G", "MinGW Makefiles",
                    "-DCMAKE_C_COMPILER=gcc.exe",
                    "-DCMAKE_CXX_COMPILER=g++.exe",
                    "-DWITH_KLEIDICV=OFF",
                    "-DWITH_CAROTENE=OFF",
                )

            classifier.startsWith("android-") -> {
                val abi = when (parts.last()) {
                    "v8a" -> "arm64-v8a"
                    "v7a" -> "armeabi-v7a"
                    else -> parts.last()
                }
                val ndk = ndkPath()
                // Android artifacts are built by the Linux CI job only; other
                // hosts may still have an NDK (GitHub runners do) but must not
                // attempt these cross builds.
                ((ndk != null) && host.isLinux) to (
                        if (ndk == null) emptyList() else listOf(
                            "-DCMAKE_TOOLCHAIN_FILE=$ndk/build/cmake/android.toolchain.cmake",
                            "-DANDROID_ABI=$abi",
                            "-DANDROID_PLATFORM=android-24",
                            "-DANDROID_STL=c++_static",
                        ) + if (abi in setOf("x86", "x86_64")) listOf(
                            "-DWITH_KLEIDICV=OFF",
                            "-DWITH_CAROTENE=OFF",
                        ) else emptyList()
                        )
            }

            else -> error("unsupported classifier '$classifier'")
        }
        val canBuildHere = buildability.first
        val extraFlags = buildability.second

        val libFile = when {
            classifier.startsWith("windows-") -> "opencv_jni.dll"
            classifier.startsWith("darwin-") -> "libopencv_jni.dylib"
            else -> "libopencv_jni.so"
        }

        val resourceDir = "cn/enaium/opencv/native/$classifier"
        val nativeOutputDir = project.layout.buildDirectory.dir("jni-native/$classifier")
        val cmakeBuildDir = project.layout.buildDirectory.dir("cmake-jni/$classifier")

        val configureJniLibrary = project.tasks.register(
            "configureJniLibrary$classifier",
            Exec::class.java,
        ) {
            group = "build"
            description = "cmake-configures libopencv_jni for $classifier."
            onlyIf { canBuildHere }
            val outDir = nativeOutputDir.get().asFile
            val buildDir = cmakeBuildDir.get().asFile
            doFirst {
                outDir.mkdirs()
                buildDir.mkdirs()
            }
            workingDir = buildDir

            // Desktop targets take JNI headers from the running JDK; Android
            // takes them from the NDK's unified sysroot.
            val jniIncludeArgs: List<String> =
                if (classifier.startsWith("android-")) {
                    val include = ndkSysrootInclude() ?: ""
                    listOf(
                        "-DCVK_JNI_INCLUDE=$include",
                        "-DCVK_JNI_INCLUDE_PLATFORM=$include",
                    )
                } else {
                    val javaHome = System.getProperty("java.home")
                        ?: System.getenv("JAVA_HOME") ?: ""
                    val platformDir = when {
                        host.isWindows -> "win32"
                        host.isMacOsX -> "darwin"
                        else -> "linux"
                    }
                    listOf(
                        "-DCVK_JNI_INCLUDE=$javaHome/include",
                        "-DCVK_JNI_INCLUDE_PLATFORM=$javaHome/include/$platformDir",
                    )
                }

            commandLine(
                listOf(
                    resolveCmakeExecutable(),
                    project.rootProject.file("native").absolutePath,
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCVK_BUILD_JNI=ON",
                    "-DCVK_JNI_BRIDGE=${project.rootProject.file("jni/jni_bridge.cpp").absolutePath}",
                    "-DCVK_OUTPUT_DIR=${outDir.absolutePath}",
                ) + jniIncludeArgs + extraFlags,
            )
        }

        val buildJniLibrary = project.tasks.register(
            "buildJniLibrary$classifier",
            Exec::class.java,
        ) {
            group = "build"
            description = "Builds $libFile for $classifier."
            onlyIf { canBuildHere }
            dependsOn(configureJniLibrary)
            workingDir = cmakeBuildDir.get().asFile
            commandLine(resolveCmakeExecutable(), "--build", ".", "--config", "Release")
            inputs.files(
                project.rootProject.file("native/CMakeLists.txt"),
                project.rootProject.file("native/cmake/merge_static.cmake"),
                project.rootProject.file("native/include/opencv_kmp.h"),
                project.rootProject.file("native/opencv_shim.cpp"),
                project.rootProject.file("jni/jni_bridge.cpp"),
            )
            inputs.dir(project.rootProject.file("opencv"))
            outputs.file(nativeOutputDir.map { it.file(libFile) })
        }

        project.tasks.named("processResources", Copy::class.java) {
            dependsOn(buildJniLibrary)
            from(buildJniLibrary.map { it.outputs.files }) {
                include(libFile)
                into(resourceDir)
            }
        }
    }

    /** Resolves cmake to an absolute path so Exec tasks survive daemon PATH drift. */
    fun resolveCmakeExecutable(): String {
        val exeName = if (OperatingSystem.current().isWindows) "cmake.exe" else "cmake"
        System.getenv("PATH")?.split(File.pathSeparator).orEmpty().forEach { dir ->
            File(dir, exeName).takeIf { it.isFile && it.canExecute() }?.let { return it.absolutePath }
        }
        listOf("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/opt/local/bin").forEach { dir ->
            File(dir, exeName).takeIf { it.isFile && it.canExecute() }?.let { return it.absolutePath }
        }
        return exeName
    }
}

