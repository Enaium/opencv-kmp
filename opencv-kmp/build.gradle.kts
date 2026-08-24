import org.gradle.internal.os.OperatingSystem
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget

plugins {
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
        }
    }

    // ==================== Native ====================
    macosArm64()
    macosX64()

    linuxX64()
    linuxArm64()

    mingwX64()

    androidNativeArm64()
    androidNativeArm32()
    androidNativeX64()
    androidNativeX86()

    // ==================== cinterop for all native targets ====================
    targets.withType<KotlinNativeTarget> {
        val targetName = this.name
        // Targets whose OpenCV build includes the KleidiCV/carotene ARM HALs.
        val armTargets = setOf(
            "macosArm64", "linuxArm64",
            "androidNativeArm64", "androidNativeArm32",
        )
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
                            "opencv_imgcodecs",
                            "opencv_features",
                            "opencv_imgproc",
                            "opencv_flann",
                            "opencv_geometry",
                            "opencv_core",
                        ) +
                                // KleidiCV HAL and carotene exist only on ARM
                                (if (targetName in armTargets) listOf(
                                    "kleidicv",
                                    "kleidicv_hal",
                                    "kleidicv_thread",
                                    "tegra_hal",
                                ) else emptyList()) +
                                listOf(
                                    "libjpeg-turbo",
                                    "libpng",
                                    "zlib",
                                )
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
                // OpenCV on the JVM comes from our own JNI shared library
                // (libopencv_jni), built from the cvk shim plus the OpenCV
                // submodule. Bundle all nine artifacts so consumers get the
                // right native binary out of the box; NativeLoader picks one
                // at runtime by os.name/os.arch (including Android).
                runtimeOnly(project(":jni-jvm-linux-x86_64"))
                runtimeOnly(project(":jni-jvm-linux-aarch64"))
                runtimeOnly(project(":jni-jvm-darwin-x86_64"))
                runtimeOnly(project(":jni-jvm-darwin-aarch64"))
                runtimeOnly(project(":jni-jvm-windows-x86_64"))
                runtimeOnly(project(":jni-jvm-android-arm64-v8a"))
                runtimeOnly(project(":jni-jvm-android-armeabi-v7a"))
                runtimeOnly(project(":jni-jvm-android-x86"))
                runtimeOnly(project(":jni-jvm-android-x86_64"))
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
    // Signing requires the -Psigning.* properties (provided by CI); plain
    // publishToMavenLocal runs stay signature-free for local iteration.
    if (project.hasProperty("signing.keyId")) {
        signAllPublications()
    }

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
