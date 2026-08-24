/*
 * Per-OS/arch JNI artifact: android-x86_64.
 * Ships libopencv_jni as a classpath resource at
 * /cn/enaium/opencv/native/android-x86_64/, which NativeLoader
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

JniModules.configure(project, "android-x86_64")

mavenPublishing {
    coordinates(
        groupId = rootProject.group.toString(),
        artifactId = "opencv-kmp-jni-jvm-android-x86_64",
        version = rootProject.version.toString(),
    )
    publishToMavenCentral(automaticRelease = true)
    // Signing requires the -Psigning.* properties (provided by CI); plain
    // publishToMavenLocal runs stay signature-free for local iteration.
    if (project.hasProperty("signing.keyId")) {
        signAllPublications()
    }

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
