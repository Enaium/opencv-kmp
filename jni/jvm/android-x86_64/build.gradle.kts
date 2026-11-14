/*
 * Per-ABI JNI artifact: android-x86_64.
 * Publishes an Android library whose AAR carries
 * jni/x86_64/libopencv_jni.so, which AGP packages into the app's lib/x86_64/
 * and NativeLoader loads through System.loadLibrary.
 */
import cn.enaium.opencv.gradle.JniModules

plugins {
    id("com.android.library")
    id("com.vanniktech.maven.publish")
}

val jniLibs = JniModules.configure(project, "android-x86_64")

android {
    namespace = "cn.enaium.opencv.jni.android_x86_64"
    compileSdk = 36
    defaultConfig {
        minSdk = 26
    }
}

// AGP 9's sourceSets DSL rejects generated jniLibs directories; the variant API
// is the supported route for a build-time produced <abi>/<lib>.
androidComponents {
    onVariants { variant ->
        variant.sources.jniLibs?.addStaticSourceDirectory(jniLibs.get().asFile.absolutePath)
    }
}

mavenPublishing {
    coordinates(
        groupId = rootProject.group.toString(),
        artifactId = "opencv-kmp-jni-jvm-android-x86_64",
        version = rootProject.version.toString(),
    )
    publishToMavenCentral(automaticRelease = true)
    // Always register the signing tasks: conditional registration silently
    // skipped them when signing.* was provided via gradle.properties on some
    // runners, failing Central validation on missing .asc files. The signing
    // plugin no-ops when no keyring is configured, so local
    // publishToMavenLocal stays fine.
    signAllPublications()

    pom {
        name.set("opencv-kmp-jni-jvm-android-x86_64")
        description.set(
            "Prebuilt JNI shared library for opencv-kmp on android-x86_64. " +
                    "Loaded automatically by NativeLoader; not intended to be depended on directly.",
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
            developer { id.set("Enaium") }
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
