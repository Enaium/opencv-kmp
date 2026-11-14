/*
 * Copyright (c) 2026 Enaium
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package cn.enaium.opencv.example.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import cn.enaium.opencv.example.Gallery
import cn.enaium.opencv.example.runBasicDemo
import cn.enaium.opencv.example.runImageProcessingDemo
import cn.enaium.opencv.example.runTutorialDemos
import cn.enaium.opencv.ColorConversionCodes
import cn.enaium.opencv.Mat
import cn.enaium.opencv.MatType
import cn.enaium.opencv.toBitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.min

/**
 * Android presentation of the `:examples:basic` demos: the same [Gallery] the
 * desktop entry point feeds to `imshow` is drawn on Compose canvases.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val outDir = cacheDir.absolutePath
        setContent {
            MaterialTheme {
                DemoGallery { gallery -> runDemos(outDir, gallery) }
            }
        }
    }
}

private fun runDemos(outDir: String, gallery: Gallery): String = buildString {
    appendLine(runBasicDemo())
    appendLine(runImageProcessingDemo(outDir, gallery))
    append(runTutorialDemos(gallery))
}

private class Picture(val title: String, val bitmap: ImageBitmap)

private sealed interface Loaded {
    data object Loading : Loaded
    class Ready(val report: String, val pictures: List<Picture>) : Loaded
    class Failed(val message: String) : Loaded
}

@Composable
private fun DemoGallery(load: (Gallery) -> String) {
    var state by remember { mutableStateOf<Loaded>(Loaded.Loading) }
    LaunchedEffect(Unit) {
        state = withContext(Dispatchers.Default) {
            runCatching {
                Gallery().use { gallery ->
                    val report = load(gallery)
                    // Bitmaps are copies, so the Mats can be released by use {}.
                    Loaded.Ready(report, gallery.images.map { Picture(it.title, it.mat.toArgbBitmap()) })
                }
            }.getOrElse { Loaded.Failed(it.message ?: it.toString()) }
        }
    }
    Surface(Modifier.fillMaxSize().safeDrawingPadding()) {
        when (val current = state) {
            Loaded.Loading -> Box(Modifier.fillMaxSize(), Alignment.Center) { CircularProgressIndicator() }
            is Loaded.Failed -> Text("demo failed: ${current.message}", Modifier.padding(24.dp))
            is Loaded.Ready -> LazyColumn(
                Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                items(current.pictures) { PictureCard(it) }
                item {
                    Text(current.report, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                }
            }
        }
    }
}

@Composable
private fun PictureCard(picture: Picture) {
    Card {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(picture.title, style = MaterialTheme.typography.titleSmall)
            val width = picture.bitmap.width.coerceAtLeast(1)
            val height = picture.bitmap.height.coerceAtLeast(1)
            Canvas(Modifier.fillMaxWidth().aspectRatio(width.toFloat() / height)) {
                // Fit the bitmap into the canvas without distortion.
                val scale = min(size.width / width, size.height / height)
                val dstWidth = (width * scale).toInt()
                val dstHeight = (height * scale).toInt()
                drawImage(
                    image = picture.bitmap,
                    dstOffset = IntOffset(
                        ((size.width - dstWidth) / 2).toInt(),
                        ((size.height - dstHeight) / 2).toInt(),
                    ),
                    dstSize = IntSize(dstWidth, dstHeight),
                )
            }
        }
    }
}

/**
 * `Mat.toBitmap` mirrors OpenCV's Utils and only accepts CV_8UC4; the demos
 * produce single-channel and BGR pictures, so convert them first.
 */
private fun Mat.toArgbBitmap(): ImageBitmap {
    var converted: Mat? = null
    try {
        val rgba = when (type) {
            MatType.CV_8UC4 -> this
            MatType.CV_8UC1 -> cvtColor(ColorConversionCodes.GRAY2RGBA).also { converted = it }
            MatType.CV_8UC3 -> cvtColor(ColorConversionCodes.BGR2RGBA).also { converted = it }
            else -> convertTo(MatType.CV_8UC4).also { converted = it }
        }
        return rgba.toBitmap().asImageBitmap()
    } finally {
        converted?.close()
    }
}
