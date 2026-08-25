package cn.enaium.opencv.gradle

import java.io.File

/**
 * Resolves the newest android.jar under $ANDROID_HOME/platforms (or
 * local.properties' sdk.dir), mirroring how JniModules locates the NDK.
 * Returns null on hosts without an Android SDK; the opencv-kmp-android module
 * is then excluded from settings.gradle.kts.
 */
fun resolveAndroidJar(): File? {
    val home = System.getenv("ANDROID_HOME")
        ?: System.getenv("ANDROID_SDK_ROOT")
        ?: File("local.properties").takeIf { it.isFile }
            ?.readLines()
            ?.firstOrNull { it.trimStart().startsWith("sdk.dir=") }
            ?.substringAfter('=')
            ?.trim()
        ?: (System.getProperty("user.home") + "/Android/Sdk")
    val platforms = File(home, "platforms")
    if (!platforms.isDirectory) return null
    return platforms.listFiles()
        ?.filter { it.isDirectory && it.name.startsWith("android-") }
        ?.maxByOrNull { it.name }
        ?.let { File(it, "android.jar") }
        ?.takeIf { it.isFile }
}
