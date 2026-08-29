import org.gradle.internal.os.OperatingSystem
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget

plugins {
    alias(libs.plugins.android.kmp.library)
    alias(libs.plugins.kotlin.multiplatform)
    id("com.vanniktech.maven.publish")
}

group = rootProject.group
version = rootProject.version

val nativeLayerDir = rootProject.file("native")

val hostOs = OperatingSystem.current()
val hostArch = System.getProperty("os.arch").lowercase()

// Apple targets build on macOS via Xcode; linuxX64 is built on Linux hosts;
// linuxArm64 is built on Linux aarch64 hosts or cross-compiled from x86_64
// with the aarch64-linux-gnu toolchain; mingwX64 is cross-compiled on Linux
// hosts with the x86_64-w64-mingw32 toolchain (Windows hosts default to MSVC,
// whose archives are incompatible with Kotlin/Native's MinGW linker).
fun hasMingwCrossToolchain(): Boolean {
    val name = "x86_64-w64-mingw32-gcc"
    return System.getenv("PATH")?.split(File.pathSeparator).orEmpty().any { dir ->
        val f = File(dir, name)
        f.isFile && f.canExecute()
    }
}

fun hasAarch64CrossToolchain(): Boolean {
    val name = "aarch64-linux-gnu-gcc"
    return System.getenv("PATH")?.split(File.pathSeparator).orEmpty().any { dir ->
        val f = File(dir, name)
        f.isFile && f.canExecute()
    }
}

// Reads sdk.dir from local.properties (the standard place Gradle's Android
// plugins put the SDK path, e.g. /Users/<user>/Library/Android/sdk).
fun localSdkDir(): String? {
    val f = rootProject.file("local.properties")
    if (!f.isFile) return null
    return f.readLines()
        .firstOrNull { it.trimStart().startsWith("sdk.dir=") }
        ?.substringAfter('=')
        ?.trim()
}

// Locates an installed Android NDK, preferring the highest version under
// $ANDROID_HOME (or $ANDROID_SDK_ROOT, or local.properties' sdk.dir, or
// ~/Android/Sdk). androidNative targets cross-compile the OpenCV static
// library with this toolchain.
fun androidNdkPath(): String? {
    val home = System.getenv("ANDROID_HOME")
        ?: System.getenv("ANDROID_SDK_ROOT")
        ?: localSdkDir()
        ?: System.getProperty("user.home") + "/Android/Sdk"
    val ndkDir = File(home, "ndk")
    if (!ndkDir.isDirectory) return null
    return ndkDir.listFiles()
        ?.filter { it.isDirectory && it.name.matches(Regex("\\d+\\.\\d+\\.\\d+.*")) }
        ?.sortedBy { it.name }
        ?.lastOrNull()
        ?.absolutePath
}

fun canBuildNativeTarget(targetName: String): Boolean {
    return when {
        hostOs.isMacOsX && targetName.startsWith("macos") -> true
        // Apple mobile klibs embed an OpenCV static library cross-compiled
        // with the Xcode toolchain, so any macOS host with Xcode can build
        // them (simulator and device slices alike).
        hostOs.isMacOsX && targetName in setOf(
            "iosArm64", "iosSimulatorArm64", "iosX64",
            "tvosArm64", "tvosSimulatorArm64",
            "watchosArm64", "watchosSimulatorArm64", "watchosDeviceArm64",
        ) -> true
        hostOs.isLinux && targetName == "linuxX64" -> true
        hostOs.isLinux && targetName == "linuxArm64" &&
                (hostArch == "aarch64" || hostArch == "arm64" || hasAarch64CrossToolchain()) -> true
        hostOs.isLinux && targetName == "mingwX64" && hasMingwCrossToolchain() -> true
        targetName.startsWith("androidNative") && androidNdkPath() != null -> true
        else -> false
    }
}

fun resolveCmakeExecutable(): String {
    val exeName = if (OperatingSystem.current().isWindows) "cmake.exe" else "cmake"

    System.getenv("PATH")?.split(File.pathSeparator).orEmpty().forEach { dir ->
        val candidate = File(dir, exeName)
        if (candidate.isFile && candidate.canExecute()) return candidate.absolutePath
    }

    val extraPaths = listOf(
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/opt/local/bin",
    )
    extraPaths.forEach { dir ->
        val candidate = File(dir, exeName)
        if (candidate.isFile && candidate.canExecute()) return candidate.absolutePath
    }

    return exeName
}

val cmakeExecutable: String by lazy { resolveCmakeExecutable() }

kotlin {
    // ==================== JVM ====================
    jvm {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_21)
        }
        testRuns["test"].executionTask.configure {
            useJUnitPlatform()
            testLogging {
                showStandardStreams = true
                showExceptions = true
                exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
            }
        }
    }

    // ==================== Android (JVM on ART) ====================
    // Same JNI runtime as the desktop JVM target; NativeLoader picks the
    // android-* classifier at runtime. Bitmap <-> Mat interop lives in
    // androidMain (android.graphics is not on the desktop classpath).
    android {
        namespace = "cn.enaium.opencv"
        compileSdk = 36
        minSdk = 26

        // Expose the host-side unit test task (testAndroidHostTest) so CI
        // can run the common test suite against the android variant's JVM
        // classes.
        withHostTest {}
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_17)
        }
    }

    // ==================== Native ====================
    macosArm64()
    macosX64()

    iosArm64()
    iosSimulatorArm64()
    iosX64()

    tvosArm64()
    tvosSimulatorArm64()

    watchosArm64()
    watchosSimulatorArm64()
    watchosDeviceArm64()

    linuxX64()
    linuxArm64()

    mingwX64()

    androidNativeArm64()
    androidNativeArm32()
    androidNativeX64()
    androidNativeX86()

    // ==================== cinterop for all native targets ====================
    // Consumers link the platform frameworks automatically: the system
    // library arguments live in opencv.def under the linkerOpts.<Family>
    // keys (see src/nativeInterop/cinterop/opencv.def), which Kotlin/Native
    // records in the cinterop klib manifest and forwards into every
    // consumer's link command - library users only need the import.
    //
    // The binaries.configureEach linkerOpts below are a fallback for this
    // module's own test/link targets so they work even when the cinterop
    // klib output is not consulted (e.g. IDE-less test runs).
    targets.withType<KotlinNativeTarget>().configureEach {
        when (name) {
            "linuxX64", "linuxArm64" -> binaries.configureEach {
                linkerOpts(
                    // GTK3 lives in the host system lib dir, which the
                    // Kotlin/Native sysroot does not search by default.
                    "-L/usr/lib/x86_64-linux-gnu",
                    "-lgtk-3", "-lgdk-3", "-lpangocairo-1.0", "-lpango-1.0",
                    "-lharfbuzz", "-latk-1.0", "-lcairo-gobject", "-lcairo",
                    "-lgdk_pixbuf-2.0", "-lgio-2.0", "-lgobject-2.0", "-lglib-2.0",
                )
            }
            "macosArm64", "macosX64" -> binaries.configureEach {
                // Cocoa/AppKit: highgui. AVFoundation/CoreMedia/CoreVideo/
                // QuartzCore: the videoio AVFoundation backend.
                linkerOpts(
                    "-framework", "Cocoa", "-framework", "AppKit",
                    "-framework", "AVFoundation", "-framework", "CoreMedia",
                    "-framework", "CoreVideo", "-framework", "QuartzCore",
                )
            }
            "iosArm64", "iosSimulatorArm64", "iosX64" -> binaries.configureEach {
                linkerOpts(
                    "-framework", "Foundation", "-framework", "CoreFoundation",
                    "-framework", "UIKit", "-framework", "ImageIO",
                    "-framework", "CoreGraphics", "-framework", "QuartzCore",
                )
            }
            "tvosArm64", "tvosSimulatorArm64" -> binaries.configureEach {
                linkerOpts(
                    "-framework", "Foundation", "-framework", "CoreFoundation",
                    "-framework", "UIKit", "-framework", "ImageIO",
                    "-framework", "CoreGraphics", "-framework", "QuartzCore",
                )
            }
            "watchosArm64", "watchosSimulatorArm64", "watchosDeviceArm64" -> binaries.configureEach {
                linkerOpts(
                    "-framework", "Foundation", "-framework", "CoreFoundation",
                    "-framework", "ImageIO", "-framework", "CoreGraphics",
                    "-framework", "QuartzCore",
                )
            }
        }
    }

    targets.withType<KotlinNativeTarget> {
        val targetName = this.name
        // KleidiCV ships for AArch64 only; carotene (tegra_hal) covers 32-bit ARM too.
        val aarch64Targets = setOf(
            "macosArm64", "linuxArm64", "androidNativeArm64",
        )
        val arm32Targets = setOf("androidNativeArm32")
        val armHalLibs = listOf("kleidicv", "kleidicv_hal", "kleidicv_thread", "tegra_hal")
        val arm32HalLibs = listOf("tegra_hal")
        val canBuild = canBuildNativeTarget(targetName)

        compilations.getByName("main") {
            cinterops {
                create("opencv") {
                    defFile(project.file("src/nativeInterop/cinterop/opencv.def"))
                    includeDirs(rootProject.file("native/include"))
                    // Shared native sources are injected straight into every
                    // target compilation instead of living in a nativeMain
                    // fragment: the hierarchy template's empty nativeMain owns
                    // a metadata compilation that cannot resolve the cinterop
                    // package, and duplicate ownership would trip K2's
                    // single-module check.
                    defaultSourceSet.kotlin.srcDir("src/nativeMain/kotlin")
                    if (canBuild) {
                        // Embed every static library (shim + OpenCV modules
                        // + bundled codecs) into the produced cinterop klib.
                        // Order matters for single-pass linkers: dependents
                        // first, core last among modules, codecs after.
                        // Targets that can't be built on this host still get
                        // bindings (for klib publishing); the archives are
                        // built and embedded when building on the matching
                        // host.
                        val outputDir = layout.buildDirectory.dir("native/$targetName").get().asFile
                        val embeddedLibs = listOf(
                            "cvk_shim",
                        ) +
                                // highgui is compiled on every target (real
                                // window backend on desktop, the built-in
                                // "NONE" stub on Android/Apple mobile, which
                                // ptcloud requires); the shim reports
                                // unavailability through no-op stubs there.
                                // SDK-parity modules, dependents first for
                                // single-pass linkers: ptcloud needs highgui
                                // + video + features, calib needs objdetect
                                // + stereo, highgui needs videoio +
                                // imgcodecs, dnn needs protobuf.
                                listOf(
                                    "opencv_ptcloud",
                                    "opencv_calib",
                                    "opencv_objdetect",
                                    "opencv_dnn",
                                    "opencv_video",
                                    "opencv_stereo",
                                    "opencv_photo",
                                    "opencv_highgui",
                                    "opencv_videoio",
                                    "opencv_imgcodecs",
                                    "opencv_features",
                                    "opencv_imgproc",
                                    "opencv_flann",
                                    "opencv_geometry",
                                    "opencv_core",
                                ) +
                                // ARM HAL archives exist only on ARM targets
                                (when {
                                    targetName in aarch64Targets -> armHalLibs
                                    targetName in arm32Targets -> arm32HalLibs
                                    else -> emptyList()
                                }) +
                                listOf(
                                    "libjpeg-turbo",
                                    "libpng",
                                    "zlib",
                                    "libprotobuf",
                                ) +
                                // mingwX64: K/N's bundled GNU runtime is too
                                // old for the host-cross-compiled archives
                                (if (targetName == "mingwX64") {
                                    listOf("stdc++_mingw_x64", "gcc_mingw_x64")
                                } else {
                                    emptyList()
                                })
                        extraOpts(
                            listOf("-libraryPath", outputDir.absolutePath) +
                                    embeddedLibs.flatMap { listOf("-staticLibrary", "lib$it.a") }
                        )
                    }
                }
            }
        }
    }
    // ==================== Source sets ====================
    sourceSets {
        // The hierarchy template creates an orphaned cross-vendor nativeMain
        // whose default srcDir would double-claim the shared native sources;
        // keep it empty - those sources are injected into each target
        // compilation directly.
        maybeCreate("nativeMain").apply {
            kotlin.setSrcDirs(emptyList<String>())
            resources.setSrcDirs(emptyList<String>())
        }
        // JVM actuals (JNI bridge, loader, platform) shared verbatim by the
        // desktop jvm() and android() targets; both compile to JVM bytecode
        // against the same libopencv_jni C ABI.
        val jvmShared = maybeCreate("jvmShared")
        jvmShared.dependsOn(getByName("commonMain"))
        getByName("jvmMain").dependsOn(jvmShared)
        getByName("androidMain").dependsOn(jvmShared)
        commonMain {
            dependencies {
                implementation(kotlin("stdlib-common"))
            }
        }

        commonTest {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }

        jvmMain {
            dependencies {
                // Desktop JVM bundles only the desktop JNI libraries;
                // android-* stay in the android variant (androidMain) so a
                // Linux CI job never has to build the Android artifacts.
                // NativeLoader picks the right desktop classifier at runtime
                // by os.name/os.arch.
                runtimeOnly(project(":jni-jvm-linux-x86_64"))
                runtimeOnly(project(":jni-jvm-linux-aarch64"))
                runtimeOnly(project(":jni-jvm-darwin-x86_64"))
                runtimeOnly(project(":jni-jvm-darwin-aarch64"))
                runtimeOnly(project(":jni-jvm-windows-x86_64"))
            }
        }

        // Android variant bundles only the android-* JNI libraries; the
        // desktop JVM variant keeps bundling every classifier so a single
        // desktop dependency runs on any host OS.
        androidMain {
            dependencies {
                runtimeOnly(project(":jni-jvm-android-arm64-v8a"))
                runtimeOnly(project(":jni-jvm-android-armeabi-v7a"))
                runtimeOnly(project(":jni-jvm-android-x86"))
                runtimeOnly(project(":jni-jvm-android-x86_64"))
            }
        }

        // Android host unit tests execute on the developer/CI desktop JVM,
        // so they need a desktop JNI library to link against. Scope it to
        // this source set only - the published AAR keeps shipping just the
        // android-* libraries.
        getByName("androidHostTest") {
            dependencies {
                runtimeOnly(project(":jni-jvm-darwin-aarch64"))
                runtimeOnly(project(":jni-jvm-linux-x86_64"))
            }
        }

        jvmTest {
            dependencies {
                implementation(libs.junit.jupiter)
                runtimeOnly(libs.junit.platform.launcher)
            }
        }
    }
}

// The cvk_ shim and the JNI bridge are not designed for concurrent calls
// from one process. Parallel test execution (JUnit Platform class
// interleaving in one worker, or multiple workers) produces
// nondeterministic native-memory corruption: the suite's CalibTest showed
// garbage distortion coefficients whenever classes ran at once, and the
// same bytes corrupted an unrelated mask mean. Configure the KMP test
// tasks through their typed executionTask API (tasks.named("jvmTest",
// Test::class) silently matches nothing - KotlinJvmTest extends
// AbstractTestTask, not Test).
kotlin {
    jvm {
        testRuns.all {
            executionTask.configure {
                maxParallelForks = 1
                forkEvery = 500
                systemProperty("junit.jupiter.execution.parallel.enabled", "false")
                systemProperty("junit.jupiter.execution.parallel.mode.default", "same_thread")
                systemProperty("junit.jupiter.execution.parallel.mode.classes.default", "same_thread")
            }
        }
    }
}


fun registerNativeBuildTasks(targetName: String, cmakeFlags: List<String> = emptyList()) {
    val outputDir = layout.buildDirectory.dir("native/$targetName").get().asFile
    val cmakeBuildDir = layout.buildDirectory.dir("cmake-$targetName").get().asFile

    val configureTask = tasks.register<Exec>("configureNative_$targetName") {
        onlyIf { canBuildNativeTarget(targetName) }
        doFirst {
            cmakeBuildDir.mkdirs()
            outputDir.mkdirs()
        }
        workingDir = cmakeBuildDir
        commandLine(
            listOf(
                cmakeExecutable, nativeLayerDir.absolutePath,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCVK_OUTPUT_DIR=${outputDir.absolutePath}",
            ) + cmakeFlags
        )
    }

    val buildTask = tasks.register<Exec>("buildNative_$targetName") {
        onlyIf { canBuildNativeTarget(targetName) }
        dependsOn(configureTask)
        workingDir = cmakeBuildDir
        commandLine(cmakeExecutable, "--build", ".", "--config", "Release")
    }

    tasks.matching {
        // The Kotlin plugin names the task cinteropOpencv<Target>; compare
        // case-insensitively so the wiring survives naming drift.
        it.name.startsWith("cinterop", ignoreCase = true) &&
                it.name.endsWith(targetName.replaceFirstChar { c -> c.uppercase() })
    }.configureEach {
        dependsOn(buildTask)
        inputs.dir(outputDir)
    }
}

if (hostOs.isMacOsX) {
    // Match Kotlin/Native 2.4's minimum (macOS 12.0) so the static library
    // objects never exceed the final binary's deployment target.
    registerNativeBuildTasks(
        "macosArm64",
        listOf(
            "-DCMAKE_OSX_ARCHITECTURES=arm64",
            "-DCMAKE_SYSTEM_PROCESSOR=arm64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0",
        ),
    )
    registerNativeBuildTasks(
        "macosX64",
        listOf(
            "-DCMAKE_OSX_ARCHITECTURES=x86_64",
            "-DCMAKE_SYSTEM_PROCESSOR=x86_64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0",
            // ARM-only acceleration HALs misdetect on x86 builds from ARM hosts
            "-DWITH_KLEIDICV=OFF",
            "-DWITH_CAROTENE=OFF",
        ),
    )

    // Apple mobile targets cross-compile OpenCV with the Xcode toolchain
    // (CMAKE_SYSTEM_NAME drives CMake's IOS/TVOS/WATCHOS platform handling;
    // the sysroot picks device vs simulator). Deployment targets match
    // Kotlin/Native's own minimums for these targets.
    // KleidiCV (AArch64 HAL) builds for the arm64 Apple slices; x86_64 and
    // the arm64_32 watch device slice disable it (carotene covers arm32).
    fun appleMobileFlags(systemName: String, sysroot: String, arch: String, min: String): List<String> =
        listOf(
            "-DCMAKE_OSX_SYSROOT=$sysroot",
            "-DCMAKE_OSX_ARCHITECTURES=$arch",
            // CMAKE_SYSTEM_NAME drives CMake's IOS/TVOS/WATCHOS platform
            // flag; without it CMake treats the sysroot as a macOS build.
            // No CMAKE_OSX_DEPLOYMENT_TARGET: setting it for these targets
            // makes CMake fall out of the TVOS/WATCHOS platform handling and
            // OpenCV then compiles its macOS-only sources.
            "-DCMAKE_SYSTEM_NAME=$systemName",
        )

    val mobileKleidiOff = listOf("-DWITH_KLEIDICV=OFF", "-DWITH_CAROTENE=OFF")

    registerNativeBuildTasks("iosArm64", appleMobileFlags("iOS", "iphoneos", "arm64", "13.0"))
    registerNativeBuildTasks("iosSimulatorArm64", appleMobileFlags("iOS", "iphonesimulator", "arm64", "13.0"))
    registerNativeBuildTasks("iosX64", appleMobileFlags("iOS", "iphonesimulator", "x86_64", "13.0") + mobileKleidiOff)
    registerNativeBuildTasks("tvosArm64", appleMobileFlags("tvOS", "appletvos", "arm64", "13.0"))
    registerNativeBuildTasks("tvosSimulatorArm64", appleMobileFlags("tvOS", "appletvsimulator", "arm64", "13.0"))
    registerNativeBuildTasks("watchosArm64", appleMobileFlags("watchOS", "watchos", "arm64", "9.0"))
    registerNativeBuildTasks("watchosSimulatorArm64", appleMobileFlags("watchOS", "watchsimulator", "arm64", "9.0"))
    registerNativeBuildTasks("watchosDeviceArm64", appleMobileFlags("watchOS", "watchos", "arm64_32", "9.0") + mobileKleidiOff)
} else if (hostOs.isLinux) {
    registerNativeBuildTasks(
        "linuxX64",
        listOf(
            "-DWITH_KLEIDICV=OFF",
            "-DWITH_CAROTENE=OFF",
        ),
    )
    // linuxArm64 is cross-compiled on x86_64 hosts with the aarch64-linux-gnu
    // toolchain (canBuildNativeTarget gates on it).
    registerNativeBuildTasks(
        "linuxArm64",
        listOf(
            "-DCMAKE_SYSTEM_NAME=Linux",
            "-DCMAKE_SYSTEM_PROCESSOR=aarch64",
            "-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc",
            "-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++",
            // GCC's outline atomics emit __aarch64_* helper calls that the
            // Kotlin/Native sysroot cannot resolve; keep atomics inline.
            "-DCMAKE_C_FLAGS=-mno-outline-atomics",
            "-DCMAKE_CXX_FLAGS=-mno-outline-atomics",
            "-DCMAKE_ASM_COMPILER=aarch64-linux-gnu-gcc",
        ),
    )
    // Cross-compile the MinGW archive with the x86_64-w64-mingw32 toolchain
    // (canBuildNativeTarget gates on it).
    registerNativeBuildTasks(
        "mingwX64",
        listOf(
            "-DCMAKE_SYSTEM_NAME=Windows",
            "-DCMAKE_SYSTEM_PROCESSOR=x86_64",
            "-DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc",
            "-DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++",
            "-DWITH_KLEIDICV=OFF",
            "-DWITH_CAROTENE=OFF",
            // the klib embeds the full GNU runtime; drop the compat shims
            "-DCVK_SKIP_STDCPP_SHIM=ON",
        ),
    )
}

// Windows-host klib builds would need an MSVC-compatible archive pipeline;
// the mingwX64 klib is produced by the Linux CI job instead.

androidNdkPath()?.let { ndk ->
    val toolchain = "$ndk/build/cmake/android.toolchain.cmake"
    val androidFlags = { abi: String ->
        listOf(
            "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
            "-DANDROID_ABI=$abi",
            "-DANDROID_PLATFORM=android-24",
            "-DANDROID_STL=c++_static",
        ) + if (abi in setOf("x86", "x86_64")) listOf(
            "-DWITH_KLEIDICV=OFF",
            "-DWITH_CAROTENE=OFF",
        ) else emptyList()
    }
    registerNativeBuildTasks("androidNativeArm64", androidFlags("arm64-v8a"))
    registerNativeBuildTasks("androidNativeArm32", androidFlags("armeabi-v7a"))
    registerNativeBuildTasks("androidNativeX64", androidFlags("x86_64"))
    registerNativeBuildTasks("androidNativeX86", androidFlags("x86"))
}

// ==================== Publishing ====================
mavenPublishing {
    publishToMavenCentral(automaticRelease = true)
    // Always register the signing tasks: conditional registration silently
    // skipped them when signing.* was provided via gradle.properties on some
    // runners, failing Central validation on missing .asc files. The signing
    // plugin no-ops when no keyring is configured, so local
    // publishToMavenLocal stays fine.
    signAllPublications()

    coordinates(
        groupId = group.toString(),
        artifactId = "opencv-kmp",
        // null -> the plugin falls back to project.version
        version = null,
    )

    pom {
        name.set("opencv-kmp")
        description.set(
            "Kotlin Multiplatform bindings for OpenCV (core, imgproc, imgcodecs). " +
                    "JVM uses a self-contained JNI shared library built from the OpenCV submodule; " +
                    "native targets embed the statically compiled OpenCV library into the published klib.",
        )
        url.set("https://github.com/Enaium/opencv-kmp")
        inceptionYear.set("2026")

        licenses {
            license {
                name.set("MIT")
                url.set("https://opensource.org/license/mit")
                distribution.set("repo")
            }
        }

        developers {
            developer {
                id.set("Enaium")
            }
        }

        scm {
            url.set("https://github.com/Enaium/opencv-kmp")
            connection.set("scm:git:git@github.com:Enaium/opencv-kmp.git")
            developerConnection.set("scm:git:git@github.com:Enaium/opencv-kmp.git")
        }

        issueManagement {
            system.set("GitHub")
            url.set("https://github.com/Enaium/opencv-kmp/issues")
        }
    }
}
