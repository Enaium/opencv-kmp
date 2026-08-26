# opencv-kmp

Kotlin Multiplatform bindings for [OpenCV](https://github.com/opencv/opencv) (5.x, `core` / `imgproc` / `imgcodecs`).

- **Kotlin/Native** — OpenCV is compiled statically per target and embedded into the published klib through cinterop; consumers link it without any extra setup.
- **JVM (desktop + Android)** — a self-contained JNI shared library (`libopencv_jni`) is built from the OpenCV submodule and shipped as a classpath resource per OS/arch; `NativeLoader` extracts and loads the matching one at runtime.

Both platforms call the same thin C ABI (`native/`, `cvk_` prefix) that wraps the C++ API, so behavior is identical everywhere. All native calls are exception-safe: failures surface as `OpenCVException` (with `cv::Exception::what()` text) or nullable returns.

## Platforms

| Target | Native lib | JVM lib |
| --- | --- | --- |
| macOS arm64 / x64 | klib-embedded static | `darwin-aarch64` / `darwin-x86_64` |
| Linux x64 / arm64 | klib-embedded static | `linux-x86_64` / `linux-aarch64` |
| Windows x64 | klib-embedded static (mingw) | `windows-x86_64` |
| Android (native) arm32 / arm64 / x86 / x86_64 | klib-embedded static | — |
| Android (JVM) armv7 / armv8 / x86 / x86_64 | — | `android-*` |

## Usage

```kotlin
import cn.enaium.opencv.*

fun main() {
    println(opencvVersion)

    mat(rows = 2, cols = 2, type = MatType.CV_32FC1, fill = Scalar.all(1.5)).use { a ->
        eye(2, 2, MatType.CV_32FC1).use { identity ->
            (a + identity).use { sum -> println(sum[0, 0]) } // 2.5
        }
        (a * 3.0).use { println(it[0, 0]) }                  // 4.5
    }

    imread("input.png")?.use { image ->
        val gray = image.cvtColor(ColorConversionCodes.BGR2GRAY)
        val edges = gray.canny(threshold1 = 50.0, threshold2 = 150.0)
        imwrite("edges.png", edges)
        gray.close()
        edges.close()
    }
}
```

Gradle:

```kotlin
dependencies {
    implementation("cn.enaium.opencv:opencv-kmp:<version>")
}
```

The JVM artifact bundles all nine JNI libraries (~9 × a few MB); Android apps get the right one automatically.

### API highlights

- Operators on `Mat`: `+ - * /` with another `Mat` or a `Scalar`, indexed `get`/`set`, `times(scale)`, broadcast `+ - / Double`, `abs()`/`squared()`, infix `bitwiseAnd/Or/Xor`, `diff`, `rsub`.
- Scalar arithmetic: `Scalar + - * / Scalar`, `* / + - Double`, `unaryMinus`, infix `dist`.
- Extensions in common code: `toGray()`, `toFloat32()`, `normalize01()`, `mirror()`, `rotate90/180/270()`, `pixels: ByteArray`, `shape`, `fill { }`.
- Factories: `mat()`, `zeros()`, `ones()`, `eye()`, `imread()`, `imwrite()`, `imencode()`, `imdecode()`.
- Operators on `Mat`: `+ - * /`, indexed `get`/`set`, `times(scale)`, infix `bitwiseAnd/Or/Xor`, infix `diff`.
- Extensions in common code: `toGray()`, `toFloat32()`, `normalize01()`, `mirror()`, `rotate90/180/270()`, `pixels: ByteArray`, `shape`, `fill { }`.
- Factories: `mat()`, `zeros()`, `ones()`, `eye()`, `imread()`, `imwrite()`, `imencode()`, `imdecode()`.

## Building from source

Requirements: JDK 21+, CMake ≥3.16, a C++17 compiler; the OpenCV sources are a git submodule (`git submodule update --init --recursive`). Cross builds additionally need:

- linuxArm64 klib / linux JNI: `aarch64-linux-gnu-gcc`
- mingwX64 klib: `x86_64-w64-mingw32-gcc`
- androidNative klibs / android JNI libs: Android NDK (`sdk.dir` in `local.properties`, `$ANDROID_HOME`, or `~/Android/Sdk`)

macOS klibs and the darwin JNI artifacts are built on macOS hosts; the windows-x86_64 JNI DLL is built on Windows hosts (MinGW via choco works).

```bash
# build + test everything buildable on this host
./gradlew :opencv-kmp:jvmTest :opencv-kmp:linuxX64Test   # platform tests
./gradlew :examples:basic:jvmRun                          # JVM demo
./gradlew :examples:basic:runDebugExecutableLinuxX64     # native demo

# publish local-platform artifacts to mavenLocal
./gradlew :opencv-kmp:publishToMavenLocal :jni-jvm-linux-x86_64:publishToMavenLocal
```

Each Kotlin/Native target drives its own CMake configuration (`configureNative_<target>` / `buildNative_<target>`) that compiles OpenCV statically (`BUILD_LIST=core,imgproc,imgcodecs`, bundled zlib/libpng/libjpeg-turbo, no network downloads) plus the cvk shim, merges all archives into one `libopencv_kmp.a`, and embeds it into the klib.

## CI

Two manually-dispatched workflows (Actions tab):

- **Test** — builds/tests every platform on matching runners and publishes to Maven Local.
- **Publish** — releases every artifact to Maven Central. The version is fixed in `build.gradle.kts`; nothing to input.

Required secrets for Publish: `MAVEN_CENTRAL_USERNAME`, `MAVEN_CENTRAL_PASSWORD`, `SIGNING_KEY` (base64 .gpg), `SIGNING_KEY_ID`, `SIGNING_PASSWORD`.

## License

MIT — see [LICENSE](LICENSE). OpenCV itself stays under its Apache 2.0 license.
