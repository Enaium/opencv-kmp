package cn.enaium.opencv.example.android

import android.app.Activity
import android.os.Bundle
import android.widget.TextView
import cn.enaium.opencv.getVersionString
import cn.enaium.opencv.mat

/** Touches the native library through the public API; any packaging mistake fails here. */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val matrix = mat(2, 3)
        val report = buildString {
            append("OpenCV ").append(getVersionString())
            append('\n').append("Mat ").append(matrix.rows).append('x').append(matrix.cols)
        }
        matrix.close()
        setContentView(TextView(this).apply { text = report })
    }
}
