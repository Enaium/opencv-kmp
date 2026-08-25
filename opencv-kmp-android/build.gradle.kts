import cn.enaium.opencv.gradle.resolveAndroidJar

plugins {
    kotlin("jvm")
    `java-library`
    alias(libs.plugins.maven.publish)
}

// Android JVM platform split: desktop consumers use :opencv-kmp's jvm target;
// this module adds the android.graphics.Bitmap interop that org.opencv.android
// ships as Utils, compiled against the SDK's android.jar (no reflection).

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

val androidJar = resolveAndroidJar()
    ?: throw GradleException(
        "opencv-kmp-android requires an Android SDK (ANDROID_HOME or " +
            "local.properties sdk.dir) to compile against android.jar",
    )

dependencies {
    compileOnly(files(androidJar))
    api(project(":opencv-kmp"))
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
    }
}

mavenPublishing {
    publishToMavenCentral(automaticRelease = true)
    if (project.hasProperty("signing.keyId")) {
        signAllPublications()
    }

    pom {
        name.set("opencv-kmp-android")
        description.set(
            "Android JVM interop for opencv-kmp: Bitmap <-> Mat conversions as " +
                "Kotlin extensions over android.graphics.Bitmap.",
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
            connection.set("scm:git:github.com/Enaium/opencv-kmp.git")
            developerConnection.set("scm:git:ssh://github.com/Enaium/opencv-kmp.git")
        }
        issueManagement {
            system.set("GitHub")
            url.set("https://github.com/Enaium/opencv-kmp/issues")
        }
    }
}
