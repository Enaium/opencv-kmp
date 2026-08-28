
pluginManagement {
    repositories {
    google()
        gradlePluginPortal()
    }
}

plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
    google()
        mavenCentral()
    }
}

rootProject.name = "opencv-kmp"

include(":opencv-kmp")
include(":examples:basic")
include(":examples:tutorials")

// Per-OS/arch JNI artifacts that bundle the prebuilt libopencv_jni shared
// library as a classpath resource. NativeLoader extracts the matching one at
// runtime.
listOf(
    "linux-x86_64",
    "linux-aarch64",
    "darwin-x86_64",
    "darwin-aarch64",
    "windows-x86_64",
    "android-arm64-v8a",
    "android-armeabi-v7a",
    "android-x86",
    "android-x86_64",
).forEach { classifier ->
    val name = ":jni-jvm-$classifier"
    include(name)
    project(name).projectDir = file("jni/jvm/$classifier")
}
