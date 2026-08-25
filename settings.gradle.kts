import java.io.File

fun resolveAndroidJar(): File? {
    val home = System.getenv("ANDROID_HOME")
        ?: System.getenv("ANDROID_SDK_ROOT")
        ?: File("local.properties").takeIf { it.isFile }
            ?.readLines()?.firstOrNull { it.trimStart().startsWith("sdk.dir=") }
            ?.substringAfter('=')?.trim()
        ?: (System.getProperty("user.home") + "/Android/Sdk")
    val platforms = File(home, "platforms")
    if (!platforms.isDirectory) return null
    return platforms.listFiles()
        ?.filter { it.isDirectory && it.name.startsWith("android-") }
        ?.maxByOrNull { it.name }
        ?.let { File(it, "android.jar") }
        ?.takeIf { it.isFile }
}

pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
        mavenLocal()
        mavenCentral()
    }
}

rootProject.name = "opencv-kmp"

include(":opencv-kmp")

include(":examples:basic")

// Android JVM interop (Bitmap <-> Mat extensions); needs an Android SDK so the
// module can compile against android.jar.
if (resolveAndroidJar() != null) {
    include(":opencv-kmp-android")
    project(":opencv-kmp-android").projectDir = file("opencv-kmp-android")
}

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
