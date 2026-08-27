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
package cn.enaium.opencv

// =========================================================================
// core value types (org.opencv.core parity)
// =========================================================================

/**
 * Structure for matching: query descriptor index, train descriptor index,
 * train image index and distance between descriptors.
 */
data class DMatch(
    val queryIdx: Int,
    val trainIdx: Int,
    val imgIdx: Int,
    val distance: Float,
) {
    /** Whether this match is closer than [other]. */
    fun lessThan(other: DMatch): Boolean = distance < other.distance
}

/**
 * Point of interest in an image: coordinates, diameter of the useful
 * adjacent area, orientation (-1 if not applicable), the selection
 * response, the pyramid octave and an object/cluster id.
 */
data class KeyPoint(
    val x: Float,
    val y: Float,
    val size: Float,
    val angle: Float,
    val response: Float,
    val octave: Int,
    val classId: Int,
)

/** Half-open index interval `[start, end)` over a Mat's index space. */
data class Range(val start: Int, val end: Int) {

    /** True when the range contains no elements (`end <= start`). */
    val empty: Boolean get() = end <= start

    /** Number of elements; 0 for an empty range. */
    fun size(): Int = if (empty) 0 else end - start

    /** Whether [p] lies inside `[start, end)`. */
    fun contains(p: Int): Boolean = start <= p && p < end

    /** The overlap of this range with [other]; empty when disjoint. */
    fun intersection(other: Range): Range {
        val s = maxOf(start, other.start)
        val e = maxOf(minOf(end, other.end), s)
        return Range(s, e)
    }

    /** Both bounds shifted by [delta]. */
    fun shift(delta: Int): Range = Range(start + delta, end + delta)

    companion object {
        /** The empty range at index 0. */
        fun empty(): Range = Range(0, 0)

        /** Every index (`Int.MIN_VALUE` until `Int.MAX_VALUE`). */
        fun all(): Range = Range(Int.MIN_VALUE, Int.MAX_VALUE)
    }
}

/** Double-precision rectangle backing ROI-style operations (`cv::Rect2d`). */
data class Rect2d(
    val x: Double,
    val y: Double,
    val width: Double,
    val height: Double,
) {
    /** Rectangle area. */
    val area: Double get() = width * height

    /** True when the rectangle covers no pixels. */
    val empty: Boolean get() = width <= 0 || height <= 0

    /** Whether [p] lies inside the half-open rectangle `[x, x+width) x [y, y+height)`. */
    fun contains(p: Point): Boolean =
        x <= p.x.toDouble() && p.x.toDouble() < x + width &&
            y <= p.y.toDouble() && p.y.toDouble() < y + height
}

/** 3D point (`cv::Point3_<double>`). */
data class Point3(val x: Double, val y: Double, val z: Double) {

    /** Dot product with [p]. */
    fun dot(p: Point3): Double = x * p.x + y * p.y + z * p.z

    /** Cross product with [p]. */
    fun cross(p: Point3): Point3 =
        Point3(
            y * p.z - z * p.y,
            z * p.x - x * p.z,
            x * p.y - y * p.x,
        )
}

/**
 * Base interface of every more-or-less complex OpenCV algorithm
 * (`cv::Algorithm`). Subclasses add their own operations and own a native
 * handle; [close] releases it.
 */
interface Algorithm : AutoCloseable {

    /** Clears the algorithm state. */
    fun clear()

    /** True when the algorithm has no internal data yet. */
    fun empty(): Boolean

    /** Saves the algorithm state to a file (derived classes must support it). */
    fun save(filename: String)

    /** The algorithm's string identifier. */
    fun getDefaultName(): String
}

/**
 * Measures passing time by counting platform ticks (`cv::TickMeter`).
 *
 * `start()`/`stop()` delimit one measured interval; every completed interval
 * increments [counter] and accumulates into [timeTicks]/[timeSum].
 */
interface TickMeter : AutoCloseable {

    /** Total accumulated ticks across stopped intervals. */
    val timeSum: Double

    /** Number of completed start/stop intervals. */
    val counter: Int

    /** Average seconds per interval (0 when nothing was measured). */
    val avgTime: Double

    /** Ticks per second of the platform clock. */
    val freq: Double

    /** Starts counting ticks. */
    fun start()

    /** Stops counting and accumulates the interval; no-op when not running. */
    fun stop()

    /** Resets the counter and accumulated time to zero. */
    fun reset()

    /** Total accumulated ticks across stopped intervals. */
    val timeTicks: Double

    /** Total accumulated time in seconds. */
    val timeSec: Double
}

/** Creates a stopped [TickMeter]. */
expect fun tickMeter(): TickMeter

// =========================================================================
// little-endian wire helpers
//
// MatOf* views and every other slice that marshals typed buffers through
// Mat.pixels read/write raw bytes with these helpers (OpenCV's native
// layouts are little-endian on every supported platform).
// =========================================================================

/** Reads a little-endian 32-bit int at byte offset [index]. */
internal fun ByteArray.readIntLE(index: Int): Int =
    (this[index].toInt() and 0xFF) or
        ((this[index + 1].toInt() and 0xFF) shl 8) or
        ((this[index + 2].toInt() and 0xFF) shl 16) or
        ((this[index + 3].toInt() and 0xFF) shl 24)

/** Writes [value] little-endian at byte offset [index]. */
internal fun ByteArray.writeIntLE(index: Int, value: Int) {
    this[index] = value.toByte()
    this[index + 1] = (value shr 8).toByte()
    this[index + 2] = (value shr 16).toByte()
    this[index + 3] = (value shr 24).toByte()
}

/** Reads a little-endian 32-bit float at byte offset [index]. */
internal fun ByteArray.readFloatLE(index: Int): Float = Float.fromBits(readIntLE(index))

/** Writes [value] little-endian at byte offset [index]. */
internal fun ByteArray.writeFloatLE(index: Int, value: Float) = writeIntLE(index, value.toRawBits())

/** Reads a little-endian 64-bit double at byte offset [index]. */
internal fun ByteArray.readDoubleLE(index: Int): Double = Double.fromBits(readLongLE(index))

/** Writes [value] little-endian at byte offset [index]. */
internal fun ByteArray.writeDoubleLE(index: Int, value: Double) = writeLongLE(index, value.toRawBits())

private fun ByteArray.readLongLE(index: Int): Long =
    (readIntLE(index).toLong() and 0xFFFFFFFFL) or
        ((readIntLE(index + 4).toLong() and 0xFFFFFFFFL) shl 32)

private fun ByteArray.writeLongLE(index: Int, value: Long) {
    writeIntLE(index, value.toInt())
    writeIntLE(index + 4, (value shr 32).toInt())
}

// =========================================================================
// MatOf* typed array views (org.opencv.core.MatOf* parity)
//
// Each view owns exactly one backing Mat laid out as an Nx1 matrix of the
// class's wire type (noted on each class). toArray()/fromArray() move the
// raw wire buffer (ByteArray/DoubleArray/FloatArray/IntArray), while
// toList()/fromList() move typed element objects. Every conversion
// allocates a fresh backing Mat and closes the previous one, so the wrapper
// stays the single owner of one Mat; close [mat] when done with it.
// =========================================================================

/** Backing store of a MatOf* view: owns one Mat and hands out its raw bytes. */
private class MatOfStore(initial: Mat) {
    var mat: Mat = initial
        private set

    val empty: Boolean get() = mat.isEmpty

    fun total(): Int = mat.total

    /** Replaces the backing Mat with [fresh], closing the previous one. */
    fun adopt(fresh: Mat) {
        val old = mat
        mat = fresh
        old.close()
    }

    /** (Re)allocates an [elemNumber] x 1 Mat of the wire type. */
    fun alloc(elemNumber: Int, depth: Int, channels: Int) {
        if (elemNumber > 0) adopt(mat(elemNumber, 1, cvMakeType(depth, channels)))
    }

    /** Raw pixel bytes of the backing Mat. */
    fun bytes(): ByteArray = mat.pixels

    /** Replaces the pixel bytes; the backing Mat must already have that size. */
    fun putBytes(data: ByteArray) {
        mat.pixels = data
    }
}

/** Throws when [mat] is non-empty and does not have the given depth/channels. */
private fun requireWireType(mat: Mat, depth: Int, channels: Int, name: String) {
    if (!mat.isEmpty && (cvDepthOf(mat.type) != depth || cvChannelsOf(mat.type) != channels)) {
        throw IllegalArgumentException(
            "Incompatible Mat for $name: expected ${cvMakeType(depth, channels).cvTypeName}, " +
                "got ${mat.type.cvTypeName}",
        )
    }
}

/**
 * Typed byte array view over a CV_8UC1 Mat (1 byte per element).
 *
 * Wire type: CV_8UC1.
 */
class MatOfByte constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_8U, 1, "MatOfByte")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_8U, 1)))

    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Byte) : this(mat(0, 0, cvMakeType(CV_8U, 1))) {
        fromArray(a)
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of elements. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] elements. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_8U, 1)

    /** Fills the backing buffer from [a]. */
    fun fromArray(a: ByteArray) {
        if (a.isEmpty()) return
        store.alloc(a.size, CV_8U, 1)
        store.putBytes(a.copyOf())
    }

    /** Writes the [length] bytes of [a] starting at [offset]. */
    fun fromArray(offset: Int, length: Int, a: ByteArray) {
        require(offset >= 0) { "offset < 0" }
        require(length >= 0 && offset + length <= a.size) { "invalid length parameter: $length" }
        if (length == 0) return
        store.alloc(length, CV_8U, 1)
        store.putBytes(a.copyOfRange(offset, offset + length))
    }

    /** The raw backing bytes. */
    fun toArray(): ByteArray = store.bytes()

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<Byte>) {
        if (l.isEmpty()) return
        fromArray(l.toByteArray())
    }

    /** The backing bytes as a list. */
    fun toList(): List<Byte> = toArray().toList()

    /** Element at [i]. */
    operator fun get(i: Int): Byte {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        return store.bytes()[i]
    }
}

/**
 * Typed double array view over a CV_64FC1 Mat (1 double per element).
 *
 * Wire type: CV_64FC1.
 */
class MatOfDouble constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_64F, 1, "MatOfDouble")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_64F, 1)))

    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Double) : this(mat(0, 0, cvMakeType(CV_64F, 1))) {
        fromArray(a)
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of elements. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] elements. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_64F, 1)

    /** Fills the backing buffer from [a]. */
    fun fromArray(a: DoubleArray) {
        if (a.isEmpty()) return
        store.alloc(a.size, CV_64F, 1)
        val bytes = ByteArray(a.size * 8)
        var b = 0
        for (v in a) {
            bytes.writeDoubleLE(b, v)
            b += 8
        }
        store.putBytes(bytes)
    }

    /** The raw backing doubles. */
    fun toArray(): DoubleArray {
        val bytes = store.bytes()
        val out = DoubleArray(bytes.size / 8)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readDoubleLE(b)
            i++
            b += 8
        }
        return out
    }

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<Double>) {
        if (l.isEmpty()) return
        fromArray(l.toDoubleArray())
    }

    /** The backing doubles as a list. */
    fun toList(): List<Double> = toArray().toList()

    /** Element at [i]. */
    operator fun get(i: Int): Double {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        return store.bytes().readDoubleLE(i * 8)
    }
}

/**
 * Typed float array view over a CV_32FC1 Mat (1 float per element).
 *
 * Wire type: CV_32FC1.
 */
class MatOfFloat constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32F, 1, "MatOfFloat")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32F, 1)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Float) : this(mat(0, 0, cvMakeType(CV_32F, 1))) {
        fromArray(a)
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of elements. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] elements. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32F, 1)

    /** Fills the backing buffer from [a]. */
    fun fromArray(a: FloatArray) {
        if (a.isEmpty()) return
        store.alloc(a.size, CV_32F, 1)
        val bytes = ByteArray(a.size * 4)
        var b = 0
        for (v in a) {
            bytes.writeFloatLE(b, v)
            b += 4
        }
        store.putBytes(bytes)
    }

    /** The raw backing floats. */
    fun toArray(): FloatArray {
        val bytes = store.bytes()
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readFloatLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<Float>) {
        if (l.isEmpty()) return
        fromArray(l.toFloatArray())
    }

    /** The backing floats as a list. */
    fun toList(): List<Float> = toArray().toList()

    /** Element at [i]. */
    operator fun get(i: Int): Float {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        return store.bytes().readFloatLE(i * 4)
    }
}

/**
 * Typed float array view over a CV_32FC4 Mat (4 floats per element).
 *
 * Wire type: CV_32FC4.
 */
class MatOfFloat4 constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32F, 4, "MatOfFloat4")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32F, 4)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Float) : this(mat(0, 0, cvMakeType(CV_32F, 4))) {
        fromArray(a)
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of elements (each 4 floats). */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] elements. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32F, 4)

    /** Fills the backing buffer from [a]; trailing values past a whole element are dropped. */
    fun fromArray(a: FloatArray) {
        if (a.isEmpty()) return
        val count = a.size / 4
        store.alloc(count, CV_32F, 4)
        val bytes = ByteArray(count * 16)
        var i = 0
        var b = 0
        while (i < count * 4) {
            bytes.writeFloatLE(b, a[i])
            bytes.writeFloatLE(b + 4, a[i + 1])
            bytes.writeFloatLE(b + 8, a[i + 2])
            bytes.writeFloatLE(b + 12, a[i + 3])
            i += 4
            b += 16
        }
        store.putBytes(bytes)
    }

    /** The raw backing floats (4 per element). */
    fun toArray(): FloatArray {
        val bytes = store.bytes()
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readFloatLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from flat [l]. */
    fun fromList(l: List<Float>) {
        if (l.isEmpty()) return
        fromArray(l.toFloatArray())
    }

    /** The backing floats as a flat list. */
    fun toList(): List<Float> = toArray().toList()

    /** Element at [i] (the 4 floats of element [i] as a flat array). */
    operator fun get(i: Int): FloatArray {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val bytes = store.bytes()
        val b = i * 16
        return floatArrayOf(
            bytes.readFloatLE(b),
            bytes.readFloatLE(b + 4),
            bytes.readFloatLE(b + 8),
            bytes.readFloatLE(b + 12),
        )
    }
}

/**
 * Typed float array view over a CV_32FC6 Mat (6 floats per element).
 *
 * Wire type: CV_32FC6.
 */
class MatOfFloat6 constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32F, 6, "MatOfFloat6")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32F, 6)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Float) : this(mat(0, 0, cvMakeType(CV_32F, 6))) {
        fromArray(a)
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of elements (each 6 floats). */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] elements. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32F, 6)

    /** Fills the backing buffer from [a]; trailing values past a whole element are dropped. */
    fun fromArray(a: FloatArray) {
        if (a.isEmpty()) return
        val count = a.size / 6
        store.alloc(count, CV_32F, 6)
        val bytes = ByteArray(count * 24)
        var i = 0
        var b = 0
        while (i < count * 6) {
            for (k in 0 until 6) bytes.writeFloatLE(b + k * 4, a[i + k])
            i += 6
            b += 24
        }
        store.putBytes(bytes)
    }

    /** The raw backing floats (6 per element). */
    fun toArray(): FloatArray {
        val bytes = store.bytes()
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readFloatLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from flat [l]. */
    fun fromList(l: List<Float>) {
        if (l.isEmpty()) return
        fromArray(l.toFloatArray())
    }

    /** The backing floats as a flat list. */
    fun toList(): List<Float> = toArray().toList()

    /** Element at [i] (the 6 floats of element [i] as a flat array). */
    operator fun get(i: Int): FloatArray {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val bytes = store.bytes()
        val b = i * 24
        return FloatArray(6) { k -> bytes.readFloatLE(b + k * 4) }
    }
}

/**
 * Typed int array view over a CV_32SC1 Mat (1 int per element).
 *
 * Wire type: CV_32SC1.
 */
class MatOfInt constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32S, 1, "MatOfInt")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32S, 1)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Int) : this(mat(0, 0, cvMakeType(CV_32S, 1))) {
        fromArray(a)
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of elements. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] elements. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32S, 1)

    /** Fills the backing buffer from [a]. */
    fun fromArray(a: IntArray) {
        if (a.isEmpty()) return
        store.alloc(a.size, CV_32S, 1)
        val bytes = ByteArray(a.size * 4)
        var b = 0
        for (v in a) {
            bytes.writeIntLE(b, v)
            b += 4
        }
        store.putBytes(bytes)
    }

    /** The raw backing ints. */
    fun toArray(): IntArray {
        val bytes = store.bytes()
        val out = IntArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readIntLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<Int>) {
        if (l.isEmpty()) return
        fromArray(l.toIntArray())
    }

    /** The backing ints as a list. */
    fun toList(): List<Int> = toArray().toList()

    /** Element at [i]. */
    operator fun get(i: Int): Int {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        return store.bytes().readIntLE(i * 4)
    }
}

/**
 * Typed int array view over a CV_32SC4 Mat (4 ints per element).
 *
 * Wire type: CV_32SC4.
 */
class MatOfInt4 constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32S, 4, "MatOfInt4")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32S, 4)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Int) : this(mat(0, 0, cvMakeType(CV_32S, 4))) {
        fromArray(a)
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of elements (each 4 ints). */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] elements. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32S, 4)

    /** Fills the backing buffer from [a]; trailing values past a whole element are dropped. */
    fun fromArray(a: IntArray) {
        if (a.isEmpty()) return
        val count = a.size / 4
        store.alloc(count, CV_32S, 4)
        val bytes = ByteArray(count * 16)
        var i = 0
        var b = 0
        while (i < count * 4) {
            bytes.writeIntLE(b, a[i])
            bytes.writeIntLE(b + 4, a[i + 1])
            bytes.writeIntLE(b + 8, a[i + 2])
            bytes.writeIntLE(b + 12, a[i + 3])
            i += 4
            b += 16
        }
        store.putBytes(bytes)
    }

    /** The raw backing ints (4 per element). */
    fun toArray(): IntArray {
        val bytes = store.bytes()
        val out = IntArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readIntLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from flat [l]. */
    fun fromList(l: List<Int>) {
        if (l.isEmpty()) return
        fromArray(l.toIntArray())
    }

    /** The backing ints as a flat list. */
    fun toList(): List<Int> = toArray().toList()

    /** Element at [i] (the 4 ints of element [i] as a flat array). */
    operator fun get(i: Int): IntArray {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val bytes = store.bytes()
        val b = i * 16
        return intArrayOf(
            bytes.readIntLE(b),
            bytes.readIntLE(b + 4),
            bytes.readIntLE(b + 8),
            bytes.readIntLE(b + 12),
        )
    }
}

/**
 * Typed [Point] view over a CV_32SC2 Mat (integer x,y pairs).
 *
 * Wire type: CV_32SC2.
 */
class MatOfPoint constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32S, 2, "MatOfPoint")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32S, 2)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Point) : this(mat(0, 0, cvMakeType(CV_32S, 2))) {
        fromList(a.toList())
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of points. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] points. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32S, 2)

    /** Fills the backing buffer from raw flat int pairs [a]. */
    fun fromArray(a: IntArray) {
        if (a.isEmpty()) return
        val count = a.size / 2
        store.alloc(count, CV_32S, 2)
        val bytes = ByteArray(count * 8)
        var i = 0
        var b = 0
        while (i < count * 2) {
            bytes.writeIntLE(b, a[i])
            bytes.writeIntLE(b + 4, a[i + 1])
            i += 2
            b += 8
        }
        store.putBytes(bytes)
    }

    /** The raw backing ints (x,y per point). */
    fun toArray(): IntArray {
        val bytes = store.bytes()
        val out = IntArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readIntLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<Point>) {
        if (l.isEmpty()) return
        store.alloc(l.size, CV_32S, 2)
        val bytes = ByteArray(l.size * 8)
        var b = 0
        for (p in l) {
            bytes.writeIntLE(b, p.x)
            bytes.writeIntLE(b + 4, p.y)
            b += 8
        }
        store.putBytes(bytes)
    }

    /** The backing points. */
    fun toList(): List<Point> {
        val bytes = store.bytes()
        val out = ArrayList<Point>(bytes.size / 8)
        var b = 0
        while (b < bytes.size) {
            out.add(Point(bytes.readIntLE(b), bytes.readIntLE(b + 4)))
            b += 8
        }
        return out
    }

    /** Point at [i]. */
    operator fun get(i: Int): Point {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val b = i * 8
        val bytes = store.bytes()
        return Point(bytes.readIntLE(b), bytes.readIntLE(b + 4))
    }
}

/**
 * Typed [Point] view over a CV_32FC2 Mat (float x,y pairs).
 *
 * Wire type: CV_32FC2. Coordinates truncate to [Point]'s Int fields when
 * read; use [toArray]/[fromArray] to keep full float precision.
 */
class MatOfPoint2f constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32F, 2, "MatOfPoint2f")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32F, 2)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Point) : this(mat(0, 0, cvMakeType(CV_32F, 2))) {
        fromList(a.toList())
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of points. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] points. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32F, 2)

    /** Fills the backing buffer from raw flat float pairs [a]. */
    fun fromArray(a: FloatArray) {
        if (a.isEmpty()) return
        val count = a.size / 2
        store.alloc(count, CV_32F, 2)
        val bytes = ByteArray(count * 8)
        var i = 0
        var b = 0
        while (i < count * 2) {
            bytes.writeFloatLE(b, a[i])
            bytes.writeFloatLE(b + 4, a[i + 1])
            i += 2
            b += 8
        }
        store.putBytes(bytes)
    }

    /** The raw backing floats (x,y per point). */
    fun toArray(): FloatArray {
        val bytes = store.bytes()
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readFloatLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<Point>) {
        if (l.isEmpty()) return
        store.alloc(l.size, CV_32F, 2)
        val bytes = ByteArray(l.size * 8)
        var b = 0
        for (p in l) {
            bytes.writeFloatLE(b, p.x.toFloat())
            bytes.writeFloatLE(b + 4, p.y.toFloat())
            b += 8
        }
        store.putBytes(bytes)
    }

    /** The backing points (float coordinates truncated to Int). */
    fun toList(): List<Point> {
        val bytes = store.bytes()
        val out = ArrayList<Point>(bytes.size / 8)
        var b = 0
        while (b < bytes.size) {
            out.add(Point(bytes.readFloatLE(b).toInt(), bytes.readFloatLE(b + 4).toInt()))
            b += 8
        }
        return out
    }

    /** Point at [i] (float coordinates truncated to Int). */
    operator fun get(i: Int): Point {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val b = i * 8
        val bytes = store.bytes()
        return Point(bytes.readFloatLE(b).toInt(), bytes.readFloatLE(b + 4).toInt())
    }
}

/**
 * Typed [Point3] view over a CV_32SC3 Mat (integer x,y,z triples).
 *
 * Wire type: CV_32SC3. Coordinates widen to [Point3]'s Double fields when
 * read; use [toArray]/[fromArray] to keep raw int precision.
 */
class MatOfPoint3 constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32S, 3, "MatOfPoint3")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32S, 3)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Point3) : this(mat(0, 0, cvMakeType(CV_32S, 3))) {
        fromList(a.toList())
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of points. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] points. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32S, 3)

    /** Fills the backing buffer from raw flat int triples [a]. */
    fun fromArray(a: IntArray) {
        if (a.isEmpty()) return
        val count = a.size / 3
        store.alloc(count, CV_32S, 3)
        val bytes = ByteArray(count * 12)
        var i = 0
        var b = 0
        while (i < count * 3) {
            bytes.writeIntLE(b, a[i])
            bytes.writeIntLE(b + 4, a[i + 1])
            bytes.writeIntLE(b + 8, a[i + 2])
            i += 3
            b += 12
        }
        store.putBytes(bytes)
    }

    /** The raw backing ints (x,y,z per point). */
    fun toArray(): IntArray {
        val bytes = store.bytes()
        val out = IntArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readIntLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from [l] (Double coordinates truncate to Int). */
    fun fromList(l: List<Point3>) {
        if (l.isEmpty()) return
        store.alloc(l.size, CV_32S, 3)
        val bytes = ByteArray(l.size * 12)
        var b = 0
        for (p in l) {
            bytes.writeIntLE(b, p.x.toInt())
            bytes.writeIntLE(b + 4, p.y.toInt())
            bytes.writeIntLE(b + 8, p.z.toInt())
            b += 12
        }
        store.putBytes(bytes)
    }

    /** The backing points (Int coordinates widened to Double). */
    fun toList(): List<Point3> {
        val bytes = store.bytes()
        val out = ArrayList<Point3>(bytes.size / 12)
        var b = 0
        while (b < bytes.size) {
            out.add(
                Point3(
                    bytes.readIntLE(b).toDouble(),
                    bytes.readIntLE(b + 4).toDouble(),
                    bytes.readIntLE(b + 8).toDouble(),
                ),
            )
            b += 12
        }
        return out
    }

    /** Point at [i] (Int coordinates widened to Double). */
    operator fun get(i: Int): Point3 {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val b = i * 12
        val bytes = store.bytes()
        return Point3(
            bytes.readIntLE(b).toDouble(),
            bytes.readIntLE(b + 4).toDouble(),
            bytes.readIntLE(b + 8).toDouble(),
        )
    }
}

/**
 * Typed [Point3] view over a CV_32FC3 Mat (float x,y,z triples).
 *
 * Wire type: CV_32FC3.
 */
class MatOfPoint3f constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32F, 3, "MatOfPoint3f")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32F, 3)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Point3) : this(mat(0, 0, cvMakeType(CV_32F, 3))) {
        fromList(a.toList())
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of points. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] points. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32F, 3)

    /** Fills the backing buffer from raw flat float triples [a]. */
    fun fromArray(a: FloatArray) {
        if (a.isEmpty()) return
        val count = a.size / 3
        store.alloc(count, CV_32F, 3)
        val bytes = ByteArray(count * 12)
        var i = 0
        var b = 0
        while (i < count * 3) {
            bytes.writeFloatLE(b, a[i])
            bytes.writeFloatLE(b + 4, a[i + 1])
            bytes.writeFloatLE(b + 8, a[i + 2])
            i += 3
            b += 12
        }
        store.putBytes(bytes)
    }

    /** The raw backing floats (x,y,z per point). */
    fun toArray(): FloatArray {
        val bytes = store.bytes()
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readFloatLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<Point3>) {
        if (l.isEmpty()) return
        store.alloc(l.size, CV_32F, 3)
        val bytes = ByteArray(l.size * 12)
        var b = 0
        for (p in l) {
            bytes.writeFloatLE(b, p.x.toFloat())
            bytes.writeFloatLE(b + 4, p.y.toFloat())
            bytes.writeFloatLE(b + 8, p.z.toFloat())
            b += 12
        }
        store.putBytes(bytes)
    }

    /** The backing points. */
    fun toList(): List<Point3> {
        val bytes = store.bytes()
        val out = ArrayList<Point3>(bytes.size / 12)
        var b = 0
        while (b < bytes.size) {
            out.add(
                Point3(
                    bytes.readFloatLE(b).toDouble(),
                    bytes.readFloatLE(b + 4).toDouble(),
                    bytes.readFloatLE(b + 8).toDouble(),
                ),
            )
            b += 12
        }
        return out
    }

    /** Point at [i]. */
    operator fun get(i: Int): Point3 {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val b = i * 12
        val bytes = store.bytes()
        return Point3(
            bytes.readFloatLE(b).toDouble(),
            bytes.readFloatLE(b + 4).toDouble(),
            bytes.readFloatLE(b + 8).toDouble(),
        )
    }
}

/**
 * Typed [Rect] view over a CV_32SC4 Mat (integer x,y,width,height quads).
 *
 * Wire type: CV_32SC4.
 */
class MatOfRect constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32S, 4, "MatOfRect")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32S, 4)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Rect) : this(mat(0, 0, cvMakeType(CV_32S, 4))) {
        fromList(a.toList())
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of rectangles. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] rectangles. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32S, 4)

    /** Fills the backing buffer from raw flat int quads [a]. */
    fun fromArray(a: IntArray) {
        if (a.isEmpty()) return
        val count = a.size / 4
        store.alloc(count, CV_32S, 4)
        val bytes = ByteArray(count * 16)
        var i = 0
        var b = 0
        while (i < count * 4) {
            bytes.writeIntLE(b, a[i])
            bytes.writeIntLE(b + 4, a[i + 1])
            bytes.writeIntLE(b + 8, a[i + 2])
            bytes.writeIntLE(b + 12, a[i + 3])
            i += 4
            b += 16
        }
        store.putBytes(bytes)
    }

    /** The raw backing ints (x,y,width,height per rect). */
    fun toArray(): IntArray {
        val bytes = store.bytes()
        val out = IntArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readIntLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<Rect>) {
        if (l.isEmpty()) return
        store.alloc(l.size, CV_32S, 4)
        val bytes = ByteArray(l.size * 16)
        var b = 0
        for (r in l) {
            bytes.writeIntLE(b, r.x)
            bytes.writeIntLE(b + 4, r.y)
            bytes.writeIntLE(b + 8, r.width)
            bytes.writeIntLE(b + 12, r.height)
            b += 16
        }
        store.putBytes(bytes)
    }

    /** The backing rectangles. */
    fun toList(): List<Rect> {
        val bytes = store.bytes()
        val out = ArrayList<Rect>(bytes.size / 16)
        var b = 0
        while (b < bytes.size) {
            out.add(
                Rect(
                    bytes.readIntLE(b),
                    bytes.readIntLE(b + 4),
                    bytes.readIntLE(b + 8),
                    bytes.readIntLE(b + 12),
                ),
            )
            b += 16
        }
        return out
    }

    /** Rect at [i]. */
    operator fun get(i: Int): Rect {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val b = i * 16
        val bytes = store.bytes()
        return Rect(
            bytes.readIntLE(b),
            bytes.readIntLE(b + 4),
            bytes.readIntLE(b + 8),
            bytes.readIntLE(b + 12),
        )
    }
}

/**
 * Typed [Rect2d] view over a CV_64FC4 Mat (double x,y,width,height quads).
 *
 * Wire type: CV_64FC4.
 */
class MatOfRect2d constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_64F, 4, "MatOfRect2d")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_64F, 4)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: Rect2d) : this(mat(0, 0, cvMakeType(CV_64F, 4))) {
        fromList(a.toList())
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of rectangles. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] rectangles. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_64F, 4)

    /** Fills the backing buffer from raw flat double quads [a]. */
    fun fromArray(a: DoubleArray) {
        if (a.isEmpty()) return
        val count = a.size / 4
        store.alloc(count, CV_64F, 4)
        val bytes = ByteArray(count * 32)
        var i = 0
        var b = 0
        while (i < count * 4) {
            bytes.writeDoubleLE(b, a[i])
            bytes.writeDoubleLE(b + 8, a[i + 1])
            bytes.writeDoubleLE(b + 16, a[i + 2])
            bytes.writeDoubleLE(b + 24, a[i + 3])
            i += 4
            b += 32
        }
        store.putBytes(bytes)
    }

    /** The raw backing doubles (x,y,width,height per rect). */
    fun toArray(): DoubleArray {
        val bytes = store.bytes()
        val out = DoubleArray(bytes.size / 8)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readDoubleLE(b)
            i++
            b += 8
        }
        return out
    }

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<Rect2d>) {
        if (l.isEmpty()) return
        store.alloc(l.size, CV_64F, 4)
        val bytes = ByteArray(l.size * 32)
        var b = 0
        for (r in l) {
            bytes.writeDoubleLE(b, r.x)
            bytes.writeDoubleLE(b + 8, r.y)
            bytes.writeDoubleLE(b + 16, r.width)
            bytes.writeDoubleLE(b + 24, r.height)
            b += 32
        }
        store.putBytes(bytes)
    }

    /** The backing rectangles. */
    fun toList(): List<Rect2d> {
        val bytes = store.bytes()
        val out = ArrayList<Rect2d>(bytes.size / 32)
        var b = 0
        while (b < bytes.size) {
            out.add(
                Rect2d(
                    bytes.readDoubleLE(b),
                    bytes.readDoubleLE(b + 8),
                    bytes.readDoubleLE(b + 16),
                    bytes.readDoubleLE(b + 24),
                ),
            )
            b += 32
        }
        return out
    }

    /** Rect at [i]. */
    operator fun get(i: Int): Rect2d {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val b = i * 32
        val bytes = store.bytes()
        return Rect2d(
            bytes.readDoubleLE(b),
            bytes.readDoubleLE(b + 8),
            bytes.readDoubleLE(b + 16),
            bytes.readDoubleLE(b + 24),
        )
    }
}

/**
 * Typed [RotatedRect] view over a CV_32FC5 Mat
 * (cx,cy,width,height,angle floats; angle in degrees).
 *
 * Wire type: CV_32FC5.
 */
class MatOfRotatedRect constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32F, 5, "MatOfRotatedRect")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32F, 5)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: RotatedRect) : this(mat(0, 0, cvMakeType(CV_32F, 5))) {
        fromList(a.toList())
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of rectangles. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] rectangles. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32F, 5)

    /** Fills the backing buffer from raw flat float quints [a]. */
    fun fromArray(a: FloatArray) {
        if (a.isEmpty()) return
        val count = a.size / 5
        store.alloc(count, CV_32F, 5)
        val bytes = ByteArray(count * 20)
        var i = 0
        var b = 0
        while (i < count * 5) {
            for (k in 0 until 5) bytes.writeFloatLE(b + k * 4, a[i + k])
            i += 5
            b += 20
        }
        store.putBytes(bytes)
    }

    /** The raw backing floats (cx,cy,width,height,angle per rect). */
    fun toArray(): FloatArray {
        val bytes = store.bytes()
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readFloatLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<RotatedRect>) {
        if (l.isEmpty()) return
        store.alloc(l.size, CV_32F, 5)
        val bytes = ByteArray(l.size * 20)
        var b = 0
        for (r in l) {
            bytes.writeFloatLE(b, r.centerX.toFloat())
            bytes.writeFloatLE(b + 4, r.centerY.toFloat())
            bytes.writeFloatLE(b + 8, r.width.toFloat())
            bytes.writeFloatLE(b + 12, r.height.toFloat())
            bytes.writeFloatLE(b + 16, r.angle.toFloat())
            b += 20
        }
        store.putBytes(bytes)
    }

    /** The backing rectangles. */
    fun toList(): List<RotatedRect> {
        val bytes = store.bytes()
        val out = ArrayList<RotatedRect>(bytes.size / 20)
        var b = 0
        while (b < bytes.size) {
            out.add(
                RotatedRect(
                    centerX = bytes.readFloatLE(b).toDouble(),
                    centerY = bytes.readFloatLE(b + 4).toDouble(),
                    width = bytes.readFloatLE(b + 8).toDouble(),
                    height = bytes.readFloatLE(b + 12).toDouble(),
                    angle = bytes.readFloatLE(b + 16).toDouble(),
                ),
            )
            b += 20
        }
        return out
    }

    /** RotatedRect at [i]. */
    operator fun get(i: Int): RotatedRect {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val b = i * 20
        val bytes = store.bytes()
        return RotatedRect(
            centerX = bytes.readFloatLE(b).toDouble(),
            centerY = bytes.readFloatLE(b + 4).toDouble(),
            width = bytes.readFloatLE(b + 8).toDouble(),
            height = bytes.readFloatLE(b + 12).toDouble(),
            angle = bytes.readFloatLE(b + 16).toDouble(),
        )
    }
}

/**
 * Typed [KeyPoint] view over a CV_32FC7 Mat
 * (x,y,size,angle,response,octave,classId floats).
 *
 * Wire type: CV_32FC7.
 */
class MatOfKeyPoint constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32F, 7, "MatOfKeyPoint")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32F, 7)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: KeyPoint) : this(mat(0, 0, cvMakeType(CV_32F, 7))) {
        fromList(a.toList())
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of keypoints. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] keypoints. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32F, 7)

    /** Fills the backing buffer from raw flat float heptads [a]. */
    fun fromArray(a: FloatArray) {
        if (a.isEmpty()) return
        val count = a.size / 7
        store.alloc(count, CV_32F, 7)
        val bytes = ByteArray(count * 28)
        var i = 0
        var b = 0
        while (i < count * 7) {
            for (k in 0 until 7) bytes.writeFloatLE(b + k * 4, a[i + k])
            i += 7
            b += 28
        }
        store.putBytes(bytes)
    }

    /** The raw backing floats (x,y,size,angle,response,octave,classId per keypoint). */
    fun toArray(): FloatArray {
        val bytes = store.bytes()
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readFloatLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<KeyPoint>) {
        if (l.isEmpty()) return
        store.alloc(l.size, CV_32F, 7)
        val bytes = ByteArray(l.size * 28)
        var b = 0
        for (k in l) {
            bytes.writeFloatLE(b, k.x)
            bytes.writeFloatLE(b + 4, k.y)
            bytes.writeFloatLE(b + 8, k.size)
            bytes.writeFloatLE(b + 12, k.angle)
            bytes.writeFloatLE(b + 16, k.response)
            bytes.writeFloatLE(b + 20, k.octave.toFloat())
            bytes.writeFloatLE(b + 24, k.classId.toFloat())
            b += 28
        }
        store.putBytes(bytes)
    }

    /** The backing keypoints. */
    fun toList(): List<KeyPoint> {
        val bytes = store.bytes()
        val out = ArrayList<KeyPoint>(bytes.size / 28)
        var b = 0
        while (b < bytes.size) {
            out.add(
                KeyPoint(
                    x = bytes.readFloatLE(b),
                    y = bytes.readFloatLE(b + 4),
                    size = bytes.readFloatLE(b + 8),
                    angle = bytes.readFloatLE(b + 12),
                    response = bytes.readFloatLE(b + 16),
                    octave = bytes.readFloatLE(b + 20).toInt(),
                    classId = bytes.readFloatLE(b + 24).toInt(),
                ),
            )
            b += 28
        }
        return out
    }

    /** KeyPoint at [i]. */
    operator fun get(i: Int): KeyPoint {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val b = i * 28
        val bytes = store.bytes()
        return KeyPoint(
            x = bytes.readFloatLE(b),
            y = bytes.readFloatLE(b + 4),
            size = bytes.readFloatLE(b + 8),
            angle = bytes.readFloatLE(b + 12),
            response = bytes.readFloatLE(b + 16),
            octave = bytes.readFloatLE(b + 20).toInt(),
            classId = bytes.readFloatLE(b + 24).toInt(),
        )
    }
}

/**
 * Typed [DMatch] view over a CV_32FC4 Mat
 * (queryIdx,trainIdx,imgIdx,distance floats).
 *
 * Wire type: CV_32FC4. Index fields travel as floats in the wire format
 * (exact up to 2^24) and truncate back to Int when read.
 */
class MatOfDMatch constructor(initial: Mat) {

    init {
        requireWireType(initial, CV_32F, 4, "MatOfDMatch")
    }


    private val store = MatOfStore(initial)

    /** The backing matrix; always the current Nx1 wire-typed Mat. */
    val mat: Mat get() = store.mat

    /** Empty wire-typed backing Mat. */
    constructor() : this(mat(0, 0, cvMakeType(CV_32F, 4)))


    /** Wraps [a] as the backing buffer. */
    constructor(vararg a: DMatch) : this(mat(0, 0, cvMakeType(CV_32F, 4))) {
        fromList(a.toList())
    }

    /** Whether the backing Mat has no elements. */
    val empty: Boolean get() = store.empty

    /** Number of matches. */
    fun total(): Int = store.total()

    /** (Re)allocates the backing Mat to [elemNumber] matches. */
    fun alloc(elemNumber: Int) = store.alloc(elemNumber, CV_32F, 4)

    /** Fills the backing buffer from raw flat float quads [a]. */
    fun fromArray(a: FloatArray) {
        if (a.isEmpty()) return
        val count = a.size / 4
        store.alloc(count, CV_32F, 4)
        val bytes = ByteArray(count * 16)
        var i = 0
        var b = 0
        while (i < count * 4) {
            bytes.writeFloatLE(b, a[i])
            bytes.writeFloatLE(b + 4, a[i + 1])
            bytes.writeFloatLE(b + 8, a[i + 2])
            bytes.writeFloatLE(b + 12, a[i + 3])
            i += 4
            b += 16
        }
        store.putBytes(bytes)
    }

    /** The raw backing floats (queryIdx,trainIdx,imgIdx,distance per match). */
    fun toArray(): FloatArray {
        val bytes = store.bytes()
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var b = 0
        while (i < out.size) {
            out[i] = bytes.readFloatLE(b)
            i++
            b += 4
        }
        return out
    }

    /** Fills the backing buffer from [l]. */
    fun fromList(l: List<DMatch>) {
        if (l.isEmpty()) return
        store.alloc(l.size, CV_32F, 4)
        val bytes = ByteArray(l.size * 16)
        var b = 0
        for (m in l) {
            bytes.writeFloatLE(b, m.queryIdx.toFloat())
            bytes.writeFloatLE(b + 4, m.trainIdx.toFloat())
            bytes.writeFloatLE(b + 8, m.imgIdx.toFloat())
            bytes.writeFloatLE(b + 12, m.distance)
            b += 16
        }
        store.putBytes(bytes)
    }

    /** The backing matches. */
    fun toList(): List<DMatch> {
        val bytes = store.bytes()
        val out = ArrayList<DMatch>(bytes.size / 16)
        var b = 0
        while (b < bytes.size) {
            out.add(
                DMatch(
                    queryIdx = bytes.readFloatLE(b).toInt(),
                    trainIdx = bytes.readFloatLE(b + 4).toInt(),
                    imgIdx = bytes.readFloatLE(b + 8).toInt(),
                    distance = bytes.readFloatLE(b + 12),
                ),
            )
            b += 16
        }
        return out
    }

    /** DMatch at [i]. */
    operator fun get(i: Int): DMatch {
        require(i in 0 until total()) { "index $i out of bounds (total=${total()})" }
        val b = i * 16
        val bytes = store.bytes()
        return DMatch(
            queryIdx = bytes.readFloatLE(b).toInt(),
            trainIdx = bytes.readFloatLE(b + 4).toInt(),
            imgIdx = bytes.readFloatLE(b + 8).toInt(),
            distance = bytes.readFloatLE(b + 12),
        )
    }
}
