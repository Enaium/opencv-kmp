plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.android.kmp.library) apply false
    alias(libs.plugins.maven.publish) apply false
}
allprojects {
    group = "cn.enaium.opencv"
    version = "1.0.1"
}

