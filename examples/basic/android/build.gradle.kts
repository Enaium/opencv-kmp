/*
 * Compose application presenting the :examples:basic demos.
 *
 * The desktop entry points of that module show the produced pictures in
 * highgui windows; on Android the same Gallery is rendered on a Compose
 * canvas instead.
 */
plugins {
    // AGP 9 compiles Kotlin itself, no org.jetbrains.kotlin.android needed.
    alias(libs.plugins.android.application)
    alias(libs.plugins.compose.compiler)
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
    buildFeatures {
        compose = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":examples:basic"))
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material3)
    implementation(libs.activity.compose)
}
