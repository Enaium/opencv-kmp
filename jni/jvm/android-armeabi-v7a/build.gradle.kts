/*
 * Per-OS/arch JNI artifact: android-armeabi-v7a.
 * Ships libopencv_jni as a classpath resource at
 * /cn/enaium/opencv/native/android-armeabi-v7a/, which NativeLoader
 * (in :opencv-kmp's jvmMain) extracts and System.load()s at runtime.
 */
import cn.enaium.opencv.gradle.JniModules

plugins {
    `java-library`
    id("com.vanniktech.maven.publish")
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

JniModules.configure(project, "android-armeabi-v7a")

mavenPublishing {
    coordinates(
        groupId = rootProject.group.toString(),
        artifactId = "opencv-kmp-jni-jvm-android-armeabi-v7a",
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
        name.set("opencv-kmp-jni-jvm-android-armeabi-v7a")
        description.set(
            "Prebuilt JNI shared library for opencv-kmp on android-armeabi-v7a. " +
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
