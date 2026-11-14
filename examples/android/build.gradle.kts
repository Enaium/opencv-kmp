/*
 * Minimal Android application consuming the library the way an app does:
 * the AAR carries libopencv_jni.so under jni/<abi>/, AGP packages it into
 * lib/<abi>/ and NativeLoader resolves it through System.loadLibrary.
 *
 * CI assembles this module and asserts the APK contains lib/<abi>/libopencv_jni.so.
 */
plugins {
    // AGP 9 compiles Kotlin itself, no org.jetbrains.kotlin.android needed.
    alias(libs.plugins.android.application)
}

android {
    namespace = "cn.enaium.opencv.example.android"
    compileSdk = 36
    defaultConfig {
        applicationId = "cn.enaium.opencv.example.android"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":opencv-kmp"))
}
