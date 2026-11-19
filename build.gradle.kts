plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.android.kmp.library) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.compose.compiler) apply false
    alias(libs.plugins.maven.publish) apply false
}
allprojects {
    group = "cn.enaium.opencv"
    // <kmp module version>-fixed.<pinned upstream OpenCV version>: the first
    // number belongs to this module, the second to the opencv submodule (the
    // submodule stays pinned to the upstream tag; bump both by hand, the same
    // scheme as the other *-kmp bindings).
    version = "1.0.2-fixed.5.1.0"
}

