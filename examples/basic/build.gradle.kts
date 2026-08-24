import java.io.File
import org.gradle.internal.os.OperatingSystem
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.plugin.KotlinSourceSet
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

// Kotlin/Native's own Android toolchain sysroot ships the NDK stub libraries
// (libEGL, libGLESv2, ...) that the OpenCV static library references at link
// time; point -L at the per-ABI directory so shared libs link cleanly.
fun konanAndroidLibDir(abi: String): String? {
    val base = System.getProperty("user.home") + "/.konan/dependencies"
    val dir = File(base)
    if (!dir.isDirectory) return null
    val sysroot = dir.listFiles(File::isDirectory)
        ?.filter { it.name.startsWith("target-sysroot") && it.name.contains(abi) }
        ?.maxByOrNull { it.name } ?: return null
    return sysroot.walkTopDown().maxDepth(3)
        .firstOrNull { it.isDirectory && it.name == abi }
        ?.absolutePath
}

kotlin {
    jvm {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_21)
        }
    }

    macosArm64 { binaries.executable() }
    macosX64 { binaries.executable() }

    linuxX64 { binaries.executable() }
    // Link-tested on Linux CI with the aarch64 cross toolchain; not runnable
    // there. The GNU C++ runtime the OpenCV archives need is embedded in the
    // klib itself (libstdc++_linux_arm64).
    linuxArm64 { binaries.executable() }

    mingwX64 { binaries.executable() }

    // Shared libraries exist purely to link-test the androidNative klibs'
    // embedded OpenCV static library on CI.
    androidNativeArm64 {
        binaries.sharedLib("main") {
            konanAndroidLibDir("arm64-v8a")?.let { linkerOpts("-L$it") }
        }
    }
    androidNativeArm32 {
        binaries.sharedLib("main") {
            konanAndroidLibDir("armeabi-v7a")?.let { linkerOpts("-L$it") }
        }
    }
    androidNativeX64 {
        binaries.sharedLib("main") {
            konanAndroidLibDir("x86_64")?.let { linkerOpts("-L$it") }
        }
    }
    androidNativeX86 {
        binaries.sharedLib("main") {
            konanAndroidLibDir("x86")?.let { linkerOpts("-L$it") }
        }
    }

    sourceSets {
        // The hierarchy template creates an orphaned cross-vendor nativeMain
        // whose default srcDir would double-claim the shared native sources;
        // keep it empty - those sources are injected into each target
        // compilation directly.
        maybeCreate("nativeMain").apply {
            kotlin.setSrcDirs(emptyList<String>())
            resources.setSrcDirs(emptyList<String>())
        }
        jvm {
            mainRun {
                mainClass = "cn.enaium.opencv.example.Main_jvmKt"
            }
        }

        commonMain {
            dependencies {
                implementation(project(":opencv-kmp"))
            }
        }
    }

    // Shared native demo code rides along in every native compilation (see
    // the matching comment in :opencv-kmp).
    targets.withType<KotlinNativeTarget>().configureEach {
        compilations.getByName("main").defaultSourceSet.kotlin
            .srcDir("src/nativeMain/kotlin")
    }
}

tasks.withType(JavaExec::class.java).configureEach {
    if (OperatingSystem.current().isMacOsX && name == "jvmRun") {
        jvmArgs("--enable-native-access=ALL-UNNAMED", "-XstartOnFirstThread")
    }
}
