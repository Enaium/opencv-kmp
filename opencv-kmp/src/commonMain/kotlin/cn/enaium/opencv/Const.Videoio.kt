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
@file:Suppress("MemberVisibilityCanBePrivate", "PropertyName")

package cn.enaium.opencv

/**
 * videoio constants. Values match the OpenCV C++ enums exactly
 * (opencv2/videoio.hpp) and the Java SDK's `Videoio` class 1:1.
 */

/** `cv::VideoCapture` generic property identifiers. */
object VideoCaptureProperties {
    const val CAP_PROP_UNKNOWN: Int = -1
    const val CAP_PROP_POS_MSEC: Int = 0
    const val CAP_PROP_POS_FRAMES: Int = 1
    const val CAP_PROP_POS_AVI_RATIO: Int = 2
    const val CAP_PROP_FRAME_WIDTH: Int = 3
    const val CAP_PROP_FRAME_HEIGHT: Int = 4
    const val CAP_PROP_FPS: Int = 5
    const val CAP_PROP_FOURCC: Int = 6
    const val CAP_PROP_FRAME_COUNT: Int = 7
    const val CAP_PROP_FORMAT: Int = 8
    const val CAP_PROP_MODE: Int = 9
    const val CAP_PROP_BRIGHTNESS: Int = 10
    const val CAP_PROP_CONTRAST: Int = 11
    const val CAP_PROP_SATURATION: Int = 12
    const val CAP_PROP_HUE: Int = 13
    const val CAP_PROP_GAIN: Int = 14
    const val CAP_PROP_EXPOSURE: Int = 15
    const val CAP_PROP_CONVERT_RGB: Int = 16
    const val CAP_PROP_WHITE_BALANCE_BLUE_U: Int = 17
    const val CAP_PROP_RECTIFICATION: Int = 18
    const val CAP_PROP_MONOCHROME: Int = 19
    const val CAP_PROP_SHARPNESS: Int = 20
    const val CAP_PROP_AUTO_EXPOSURE: Int = 21
    const val CAP_PROP_GAMMA: Int = 22
    const val CAP_PROP_TEMPERATURE: Int = 23
    const val CAP_PROP_TRIGGER: Int = 24
    const val CAP_PROP_TRIGGER_DELAY: Int = 25
    const val CAP_PROP_WHITE_BALANCE_RED_V: Int = 26
    const val CAP_PROP_ZOOM: Int = 27
    const val CAP_PROP_FOCUS: Int = 28
    const val CAP_PROP_GUID: Int = 29
    const val CAP_PROP_ISO_SPEED: Int = 30
    const val CAP_PROP_BACKLIGHT: Int = 32
    const val CAP_PROP_PAN: Int = 33
    const val CAP_PROP_TILT: Int = 34
    const val CAP_PROP_ROLL: Int = 35
    const val CAP_PROP_IRIS: Int = 36
    const val CAP_PROP_SETTINGS: Int = 37
    const val CAP_PROP_BUFFERSIZE: Int = 38
    const val CAP_PROP_AUTOFOCUS: Int = 39
    const val CAP_PROP_SAR_NUM: Int = 40
    const val CAP_PROP_SAR_DEN: Int = 41
    const val CAP_PROP_BACKEND: Int = 42
    const val CAP_PROP_CHANNEL: Int = 43
    const val CAP_PROP_AUTO_WB: Int = 44
    const val CAP_PROP_WB_TEMPERATURE: Int = 45
    const val CAP_PROP_CODEC_PIXEL_FORMAT: Int = 46
    const val CAP_PROP_BITRATE: Int = 47
    const val CAP_PROP_ORIENTATION_META: Int = 48
    const val CAP_PROP_ORIENTATION_AUTO: Int = 49
    const val CAP_PROP_HW_ACCELERATION: Int = 50
    const val CAP_PROP_HW_DEVICE: Int = 51
    const val CAP_PROP_HW_ACCELERATION_USE_OPENCL: Int = 52
    const val CAP_PROP_OPEN_TIMEOUT_MSEC: Int = 53
    const val CAP_PROP_READ_TIMEOUT_MSEC: Int = 54
    const val CAP_PROP_STREAM_OPEN_TIME_USEC: Int = 55
    const val CAP_PROP_VIDEO_TOTAL_CHANNELS: Int = 56
    const val CAP_PROP_VIDEO_STREAM: Int = 57
    const val CAP_PROP_AUDIO_STREAM: Int = 58
    const val CAP_PROP_AUDIO_POS: Int = 59
    const val CAP_PROP_AUDIO_SHIFT_NSEC: Int = 60
    const val CAP_PROP_AUDIO_DATA_DEPTH: Int = 61
    const val CAP_PROP_AUDIO_SAMPLES_PER_SECOND: Int = 62
    const val CAP_PROP_AUDIO_BASE_INDEX: Int = 63
    const val CAP_PROP_AUDIO_TOTAL_CHANNELS: Int = 64
    const val CAP_PROP_AUDIO_TOTAL_STREAMS: Int = 65
    const val CAP_PROP_AUDIO_SYNCHRONIZE: Int = 66
    const val CAP_PROP_LRF_HAS_KEY_FRAME: Int = 67
    const val CAP_PROP_CODEC_EXTRADATA_INDEX: Int = 68
    const val CAP_PROP_FRAME_TYPE: Int = 69
    const val CAP_PROP_N_THREADS: Int = 70
    const val CAP_PROP_PTS: Int = 71
    const val CAP_PROP_DTS_DELAY: Int = 72
    const val CAP_PROP_IMAGE_SEQ_START: Int = 73
}

/** `cv::VideoWriter` generic property identifiers. */
object VideoWriterProperties {
    const val VIDEOWRITER_PROP_UNKNOWN: Int = -1
    const val VIDEOWRITER_PROP_QUALITY: Int = 1
    const val VIDEOWRITER_PROP_FRAMEBYTES: Int = 2
    const val VIDEOWRITER_PROP_NSTRIPES: Int = 3
    const val VIDEOWRITER_PROP_IS_COLOR: Int = 4
    const val VIDEOWRITER_PROP_DEPTH: Int = 5
    const val VIDEOWRITER_PROP_HW_ACCELERATION: Int = 6
    const val VIDEOWRITER_PROP_HW_DEVICE: Int = 7
    const val VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL: Int = 8
    const val VIDEOWRITER_PROP_RAW_VIDEO: Int = 9
    const val VIDEOWRITER_PROP_KEY_INTERVAL: Int = 10
    const val VIDEOWRITER_PROP_KEY_FLAG: Int = 11
    const val VIDEOWRITER_PROP_PTS: Int = 12
    const val VIDEOWRITER_PROP_DTS_DELAY: Int = 13
    const val VIDEOWRITER_PROP_COLOR_SPACE: Int = 14
    const val VIDEOWRITER_PROP_ENABLE_ALPHA: Int = 15
}

/** `cv::VideoCaptureAPIs` — backend ids for capture and writer open calls. */
object VideoCaptureAPIs {
    const val CAP_ANY: Int = 0
    const val CAP_V4L: Int = 200
    const val CAP_V4L2: Int = CAP_V4L
    const val CAP_FIREWIRE: Int = 300
    const val CAP_FIREWARE: Int = CAP_FIREWIRE
    const val CAP_IEEE1394: Int = CAP_FIREWIRE
    const val CAP_DC1394: Int = CAP_FIREWIRE
    const val CAP_CMU1394: Int = CAP_FIREWIRE
    const val CAP_DSHOW: Int = 700
    const val CAP_PVAPI: Int = 800
    const val CAP_ANDROID: Int = 1000
    const val CAP_XIAPI: Int = 1100
    const val CAP_AVFOUNDATION: Int = 1200
    const val CAP_MSMF: Int = 1400
    const val CAP_WINRT: Int = 1410
    const val CAP_INTELPERC: Int = 1500
    const val CAP_REALSENSE: Int = 1500
    const val CAP_OPENNI2: Int = 1600
    const val CAP_OPENNI2_ASUS: Int = 1610
    const val CAP_OPENNI2_ASTRA: Int = 1620
    const val CAP_GPHOTO2: Int = 1700
    const val CAP_GSTREAMER: Int = 1800
    const val CAP_FFMPEG: Int = 1900
    const val CAP_IMAGES: Int = 2000
    const val CAP_ARAVIS: Int = 2100
    const val CAP_OPENCV_MJPEG: Int = 2200
    const val CAP_INTEL_MFX: Int = 2300
    const val CAP_XINE: Int = 2400
    const val CAP_UEYE: Int = 2500
    const val CAP_OBSENSOR: Int = 2600
}

/** `cv::VideoAccelerationType` — values for CAP_PROP_HW_ACCELERATION. */
object VideoAccelerationType {
    const val VIDEO_ACCELERATION_NONE: Int = 0
    const val VIDEO_ACCELERATION_ANY: Int = 1
    const val VIDEO_ACCELERATION_D3D11: Int = 2
    const val VIDEO_ACCELERATION_VAAPI: Int = 3
    const val VIDEO_ACCELERATION_MFX: Int = 4
    const val VIDEO_ACCELERATION_DRM: Int = 5
}

/** `cv::VideoCaptureOBSensorDataType`. */
object VideoCaptureOBSensorDataType {
    const val CAP_OBSENSOR_DEPTH_MAP: Int = 0
    const val CAP_OBSENSOR_BGR_IMAGE: Int = 1
    const val CAP_OBSENSOR_IR_IMAGE: Int = 2
}

/** `cv::VideoCaptureOBSensorGenerators`. */
object VideoCaptureOBSensorGenerators {
    const val CAP_OBSENSOR_DEPTH_GENERATOR: Int = 1 shl 29
    const val CAP_OBSENSOR_IMAGE_GENERATOR: Int = 1 shl 28
    const val CAP_OBSENSOR_IR_GENERATOR: Int = 1 shl 27
    const val CAP_OBSENSOR_GENERATORS_MASK: Int =
        CAP_OBSENSOR_DEPTH_GENERATOR + CAP_OBSENSOR_IMAGE_GENERATOR + CAP_OBSENSOR_IR_GENERATOR
}

/** `cv::VideoCaptureOBSensorProperties`. */
object VideoCaptureOBSensorProperties {
    const val CAP_PROP_OBSENSOR_INTRINSIC_FX: Int = 26001
    const val CAP_PROP_OBSENSOR_INTRINSIC_FY: Int = 26002
    const val CAP_PROP_OBSENSOR_INTRINSIC_CX: Int = 26003
    const val CAP_PROP_OBSENSOR_INTRINSIC_CY: Int = 26004
    const val CAP_PROP_OBSENSOR_RGB_POS_MSEC: Int = 26005
    const val CAP_PROP_OBSENSOR_DEPTH_POS_MSEC: Int = 26006
    const val CAP_PROP_OBSENSOR_DEPTH_WIDTH: Int = 26007
    const val CAP_PROP_OBSENSOR_DEPTH_HEIGHT: Int = 26008
    const val CAP_PROP_OBSENSOR_DEPTH_FPS: Int = 26009
    const val CAP_PROP_OBSENSOR_COLOR_DISTORTION_K1: Int = 26010
    const val CAP_PROP_OBSENSOR_COLOR_DISTORTION_K2: Int = 26011
    const val CAP_PROP_OBSENSOR_COLOR_DISTORTION_K3: Int = 26012
    const val CAP_PROP_OBSENSOR_COLOR_DISTORTION_K4: Int = 26013
    const val CAP_PROP_OBSENSOR_COLOR_DISTORTION_K5: Int = 26014
    const val CAP_PROP_OBSENSOR_COLOR_DISTORTION_K6: Int = 26015
    const val CAP_PROP_OBSENSOR_COLOR_DISTORTION_P1: Int = 26016
    const val CAP_PROP_OBSENSOR_COLOR_DISTORTION_P2: Int = 26017
}

/**
 * `cv::videoio_flags_others` — legacy/backend-specific capture flags exposed
 * by the Java SDK as one flat constant block.
 */
object VideoioFlags {
    const val CAP_PROP_DC1394_OFF: Int = -4
    const val CAP_PROP_DC1394_MODE_MANUAL: Int = -3
    const val CAP_PROP_DC1394_MODE_AUTO: Int = -2
    const val CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO: Int = -1
    const val CAP_PROP_DC1394_MAX: Int = 31
    const val CAP_OPENNI_DEPTH_GENERATOR: Int = 1 shl 31
    const val CAP_OPENNI_IMAGE_GENERATOR: Int = 1 shl 30
    const val CAP_OPENNI_IR_GENERATOR: Int = 1 shl 29
    const val CAP_OPENNI_GENERATORS_MASK: Int =
        CAP_OPENNI_DEPTH_GENERATOR + CAP_OPENNI_IMAGE_GENERATOR + CAP_OPENNI_IR_GENERATOR
    const val CAP_PROP_OPENNI_OUTPUT_MODE: Int = 100
    const val CAP_PROP_OPENNI_FRAME_MAX_DEPTH: Int = 101
    const val CAP_PROP_OPENNI_BASELINE: Int = 102
    const val CAP_PROP_OPENNI_FOCAL_LENGTH: Int = 103
    const val CAP_PROP_OPENNI_REGISTRATION: Int = 104
    const val CAP_PROP_OPENNI_REGISTRATION_ON: Int = CAP_PROP_OPENNI_REGISTRATION
    const val CAP_PROP_OPENNI_APPROX_FRAME_SYNC: Int = 105
    const val CAP_PROP_OPENNI_MAX_BUFFER_SIZE: Int = 106
    const val CAP_PROP_OPENNI_CIRCLE_BUFFER: Int = 107
    const val CAP_PROP_OPENNI_MAX_TIME_DURATION: Int = 108
    const val CAP_PROP_OPENNI_GENERATOR_PRESENT: Int = 109
    const val CAP_PROP_OPENNI2_SYNC: Int = 110
    const val CAP_PROP_OPENNI2_MIRROR: Int = 111
    const val CAP_OPENNI_IMAGE_GENERATOR_PRESENT: Int =
        CAP_OPENNI_IMAGE_GENERATOR + CAP_PROP_OPENNI_GENERATOR_PRESENT
    const val CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE: Int =
        CAP_OPENNI_IMAGE_GENERATOR + CAP_PROP_OPENNI_OUTPUT_MODE
    const val CAP_OPENNI_DEPTH_GENERATOR_PRESENT: Int =
        CAP_OPENNI_DEPTH_GENERATOR + CAP_PROP_OPENNI_GENERATOR_PRESENT
    const val CAP_OPENNI_DEPTH_GENERATOR_BASELINE: Int =
        CAP_OPENNI_DEPTH_GENERATOR + CAP_PROP_OPENNI_BASELINE
    const val CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH: Int =
        CAP_OPENNI_DEPTH_GENERATOR + CAP_PROP_OPENNI_FOCAL_LENGTH
    const val CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION: Int =
        CAP_OPENNI_DEPTH_GENERATOR + CAP_PROP_OPENNI_REGISTRATION
    const val CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION_ON: Int =
        CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION
    const val CAP_OPENNI_IR_GENERATOR_PRESENT: Int =
        CAP_OPENNI_IR_GENERATOR + CAP_PROP_OPENNI_GENERATOR_PRESENT
    const val CAP_OPENNI_DEPTH_MAP: Int = 0
    const val CAP_OPENNI_POINT_CLOUD_MAP: Int = 1
    const val CAP_OPENNI_DISPARITY_MAP: Int = 2
    const val CAP_OPENNI_DISPARITY_MAP_32F: Int = 3
    const val CAP_OPENNI_VALID_DEPTH_MASK: Int = 4
    const val CAP_OPENNI_BGR_IMAGE: Int = 5
    const val CAP_OPENNI_GRAY_IMAGE: Int = 6
    const val CAP_OPENNI_IR_IMAGE: Int = 7
    const val CAP_OPENNI_VGA_30HZ: Int = 0
    const val CAP_OPENNI_SXGA_15HZ: Int = 1
    const val CAP_OPENNI_SXGA_30HZ: Int = 2
    const val CAP_OPENNI_QVGA_30HZ: Int = 3
    const val CAP_OPENNI_QVGA_60HZ: Int = 4
    const val CAP_PROP_GSTREAMER_QUEUE_LENGTH: Int = 200
    const val CAP_PROP_PVAPI_MULTICASTIP: Int = 300
    const val CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE: Int = 301
    const val CAP_PROP_PVAPI_DECIMATIONHORIZONTAL: Int = 302
    const val CAP_PROP_PVAPI_DECIMATIONVERTICAL: Int = 303
    const val CAP_PROP_PVAPI_BINNINGX: Int = 304
    const val CAP_PROP_PVAPI_BINNINGY: Int = 305
    const val CAP_PROP_PVAPI_PIXELFORMAT: Int = 306
    const val CAP_PVAPI_FSTRIGMODE_FREERUN: Int = 0
    const val CAP_PVAPI_FSTRIGMODE_SYNCIN1: Int = 1
    const val CAP_PVAPI_FSTRIGMODE_SYNCIN2: Int = 2
    const val CAP_PVAPI_FSTRIGMODE_FIXEDRATE: Int = 3
    const val CAP_PVAPI_FSTRIGMODE_SOFTWARE: Int = 4
    const val CAP_PVAPI_DECIMATION_OFF: Int = 1
    const val CAP_PVAPI_DECIMATION_2OUTOF4: Int = 2
    const val CAP_PVAPI_DECIMATION_2OUTOF8: Int = 4
    const val CAP_PVAPI_DECIMATION_2OUTOF16: Int = 8
    const val CAP_PVAPI_PIXELFORMAT_MONO8: Int = 1
    const val CAP_PVAPI_PIXELFORMAT_MONO16: Int = 2
    const val CAP_PVAPI_PIXELFORMAT_BAYER8: Int = 3
    const val CAP_PVAPI_PIXELFORMAT_BAYER16: Int = 4
    const val CAP_PVAPI_PIXELFORMAT_RGB24: Int = 5
    const val CAP_PVAPI_PIXELFORMAT_BGR24: Int = 6
    const val CAP_PVAPI_PIXELFORMAT_RGBA32: Int = 7
    const val CAP_PVAPI_PIXELFORMAT_BGRA32: Int = 8
    const val CAP_PROP_XI_DOWNSAMPLING: Int = 400
    const val CAP_PROP_XI_DATA_FORMAT: Int = 401
    const val CAP_PROP_XI_OFFSET_X: Int = 402
    const val CAP_PROP_XI_OFFSET_Y: Int = 403
    const val CAP_PROP_XI_TRG_SOURCE: Int = 404
    const val CAP_PROP_XI_TRG_SOFTWARE: Int = 405
    const val CAP_PROP_XI_GPI_SELECTOR: Int = 406
    const val CAP_PROP_XI_GPI_MODE: Int = 407
    const val CAP_PROP_XI_GPI_LEVEL: Int = 408
    const val CAP_PROP_XI_GPO_SELECTOR: Int = 409
    const val CAP_PROP_XI_GPO_MODE: Int = 410
    const val CAP_PROP_XI_LED_SELECTOR: Int = 411
    const val CAP_PROP_XI_LED_MODE: Int = 412
    const val CAP_PROP_XI_MANUAL_WB: Int = 413
    const val CAP_PROP_XI_AUTO_WB: Int = 414
    const val CAP_PROP_XI_AEAG: Int = 415
    const val CAP_PROP_XI_EXP_PRIORITY: Int = 416
    const val CAP_PROP_XI_AE_MAX_LIMIT: Int = 417
    const val CAP_PROP_XI_AG_MAX_LIMIT: Int = 418
    const val CAP_PROP_XI_AEAG_LEVEL: Int = 419
    const val CAP_PROP_XI_TIMEOUT: Int = 420
    const val CAP_PROP_XI_EXPOSURE: Int = 421
    const val CAP_PROP_XI_EXPOSURE_BURST_COUNT: Int = 422
    const val CAP_PROP_XI_GAIN_SELECTOR: Int = 423
    const val CAP_PROP_XI_GAIN: Int = 424
    const val CAP_PROP_XI_DOWNSAMPLING_TYPE: Int = 426
    const val CAP_PROP_XI_BINNING_SELECTOR: Int = 427
    const val CAP_PROP_XI_BINNING_VERTICAL: Int = 428
    const val CAP_PROP_XI_BINNING_HORIZONTAL: Int = 429
    const val CAP_PROP_XI_BINNING_PATTERN: Int = 430
    const val CAP_PROP_XI_DECIMATION_SELECTOR: Int = 431
    const val CAP_PROP_XI_DECIMATION_VERTICAL: Int = 432
    const val CAP_PROP_XI_DECIMATION_HORIZONTAL: Int = 433
    const val CAP_PROP_XI_DECIMATION_PATTERN: Int = 434
    const val CAP_PROP_XI_TEST_PATTERN_GENERATOR_SELECTOR: Int = 587
    const val CAP_PROP_XI_TEST_PATTERN: Int = 588
    const val CAP_PROP_XI_IMAGE_DATA_FORMAT: Int = 435
    const val CAP_PROP_XI_SHUTTER_TYPE: Int = 436
    const val CAP_PROP_XI_SENSOR_TAPS: Int = 437
    const val CAP_PROP_XI_AEAG_ROI_OFFSET_X: Int = 439
    const val CAP_PROP_XI_AEAG_ROI_OFFSET_Y: Int = 440
    const val CAP_PROP_XI_AEAG_ROI_WIDTH: Int = 441
    const val CAP_PROP_XI_AEAG_ROI_HEIGHT: Int = 442
    const val CAP_PROP_XI_BPC: Int = 445
    const val CAP_PROP_XI_WB_KR: Int = 448
    const val CAP_PROP_XI_WB_KG: Int = 449
    const val CAP_PROP_XI_WB_KB: Int = 450
    const val CAP_PROP_XI_WIDTH: Int = 451
    const val CAP_PROP_XI_HEIGHT: Int = 452
    const val CAP_PROP_XI_REGION_SELECTOR: Int = 589
    const val CAP_PROP_XI_REGION_MODE: Int = 595
    const val CAP_PROP_XI_LIMIT_BANDWIDTH: Int = 459
    const val CAP_PROP_XI_SENSOR_DATA_BIT_DEPTH: Int = 460
    const val CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH: Int = 461
    const val CAP_PROP_XI_IMAGE_DATA_BIT_DEPTH: Int = 462
    const val CAP_PROP_XI_OUTPUT_DATA_PACKING: Int = 463
    const val CAP_PROP_XI_OUTPUT_DATA_PACKING_TYPE: Int = 464
    const val CAP_PROP_XI_IS_COOLED: Int = 465
    const val CAP_PROP_XI_COOLING: Int = 466
    const val CAP_PROP_XI_TARGET_TEMP: Int = 467
    const val CAP_PROP_XI_CHIP_TEMP: Int = 468
    const val CAP_PROP_XI_HOUS_TEMP: Int = 469
    const val CAP_PROP_XI_HOUS_BACK_SIDE_TEMP: Int = 590
    const val CAP_PROP_XI_SENSOR_BOARD_TEMP: Int = 596
    const val CAP_PROP_XI_CMS: Int = 470
    const val CAP_PROP_XI_APPLY_CMS: Int = 471
    const val CAP_PROP_XI_IMAGE_IS_COLOR: Int = 474
    const val CAP_PROP_XI_COLOR_FILTER_ARRAY: Int = 475
    const val CAP_PROP_XI_GAMMAY: Int = 476
    const val CAP_PROP_XI_GAMMAC: Int = 477
    const val CAP_PROP_XI_SHARPNESS: Int = 478
    const val CAP_PROP_XI_CC_MATRIX_00: Int = 479
    const val CAP_PROP_XI_CC_MATRIX_01: Int = 480
    const val CAP_PROP_XI_CC_MATRIX_02: Int = 481
    const val CAP_PROP_XI_CC_MATRIX_03: Int = 482
    const val CAP_PROP_XI_CC_MATRIX_10: Int = 483
    const val CAP_PROP_XI_CC_MATRIX_11: Int = 484
    const val CAP_PROP_XI_CC_MATRIX_12: Int = 485
    const val CAP_PROP_XI_CC_MATRIX_13: Int = 486
    const val CAP_PROP_XI_CC_MATRIX_20: Int = 487
    const val CAP_PROP_XI_CC_MATRIX_21: Int = 488
    const val CAP_PROP_XI_CC_MATRIX_22: Int = 489
    const val CAP_PROP_XI_CC_MATRIX_23: Int = 490
    const val CAP_PROP_XI_CC_MATRIX_30: Int = 491
    const val CAP_PROP_XI_CC_MATRIX_31: Int = 492
    const val CAP_PROP_XI_CC_MATRIX_32: Int = 493
    const val CAP_PROP_XI_CC_MATRIX_33: Int = 494
    const val CAP_PROP_XI_DEFAULT_CC_MATRIX: Int = 495
    const val CAP_PROP_XI_TRG_SELECTOR: Int = 498
    const val CAP_PROP_XI_ACQ_FRAME_BURST_COUNT: Int = 499
    const val CAP_PROP_XI_DEBOUNCE_EN: Int = 507
    const val CAP_PROP_XI_DEBOUNCE_T0: Int = 508
    const val CAP_PROP_XI_DEBOUNCE_T1: Int = 509
    const val CAP_PROP_XI_DEBOUNCE_POL: Int = 510
    const val CAP_PROP_XI_LENS_MODE: Int = 511
    const val CAP_PROP_XI_LENS_APERTURE_VALUE: Int = 512
    const val CAP_PROP_XI_LENS_FOCUS_MOVEMENT_VALUE: Int = 513
    const val CAP_PROP_XI_LENS_FOCUS_MOVE: Int = 514
    const val CAP_PROP_XI_LENS_FOCUS_DISTANCE: Int = 515
    const val CAP_PROP_XI_LENS_FOCAL_LENGTH: Int = 516
    const val CAP_PROP_XI_LENS_FEATURE_SELECTOR: Int = 517
    const val CAP_PROP_XI_LENS_FEATURE: Int = 518
    const val CAP_PROP_XI_DEVICE_MODEL_ID: Int = 521
    const val CAP_PROP_XI_DEVICE_SN: Int = 522
    const val CAP_PROP_XI_IMAGE_DATA_FORMAT_RGB32_ALPHA: Int = 529
    const val CAP_PROP_XI_IMAGE_PAYLOAD_SIZE: Int = 530
    const val CAP_PROP_XI_TRANSPORT_PIXEL_FORMAT: Int = 531
    const val CAP_PROP_XI_SENSOR_CLOCK_FREQ_HZ: Int = 532
    const val CAP_PROP_XI_SENSOR_CLOCK_FREQ_INDEX: Int = 533
    const val CAP_PROP_XI_SENSOR_OUTPUT_CHANNEL_COUNT: Int = 534
    const val CAP_PROP_XI_FRAMERATE: Int = 535
    const val CAP_PROP_XI_COUNTER_SELECTOR: Int = 536
    const val CAP_PROP_XI_COUNTER_VALUE: Int = 537
    const val CAP_PROP_XI_ACQ_TIMING_MODE: Int = 538
    const val CAP_PROP_XI_AVAILABLE_BANDWIDTH: Int = 539
    const val CAP_PROP_XI_BUFFER_POLICY: Int = 540
    const val CAP_PROP_XI_LUT_EN: Int = 541
    const val CAP_PROP_XI_LUT_INDEX: Int = 542
    const val CAP_PROP_XI_LUT_VALUE: Int = 543
    const val CAP_PROP_XI_TRG_DELAY: Int = 544
    const val CAP_PROP_XI_TS_RST_MODE: Int = 545
    const val CAP_PROP_XI_TS_RST_SOURCE: Int = 546
    const val CAP_PROP_XI_IS_DEVICE_EXIST: Int = 547
    const val CAP_PROP_XI_ACQ_BUFFER_SIZE: Int = 548
    const val CAP_PROP_XI_ACQ_BUFFER_SIZE_UNIT: Int = 549
    const val CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_SIZE: Int = 550
    const val CAP_PROP_XI_BUFFERS_QUEUE_SIZE: Int = 551
    const val CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_COMMIT: Int = 552
    const val CAP_PROP_XI_RECENT_FRAME: Int = 553
    const val CAP_PROP_XI_DEVICE_RESET: Int = 554
    const val CAP_PROP_XI_COLUMN_FPN_CORRECTION: Int = 555
    const val CAP_PROP_XI_ROW_FPN_CORRECTION: Int = 591
    const val CAP_PROP_XI_SENSOR_MODE: Int = 558
    const val CAP_PROP_XI_HDR: Int = 559
    const val CAP_PROP_XI_HDR_KNEEPOINT_COUNT: Int = 560
    const val CAP_PROP_XI_HDR_T1: Int = 561
    const val CAP_PROP_XI_HDR_T2: Int = 562
    const val CAP_PROP_XI_KNEEPOINT1: Int = 563
    const val CAP_PROP_XI_KNEEPOINT2: Int = 564
    const val CAP_PROP_XI_IMAGE_BLACK_LEVEL: Int = 565
    const val CAP_PROP_XI_HW_REVISION: Int = 571
    const val CAP_PROP_XI_DEBUG_LEVEL: Int = 572
    const val CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION: Int = 573
    const val CAP_PROP_XI_FFS_FILE_ID: Int = 594
    const val CAP_PROP_XI_FFS_FILE_SIZE: Int = 580
    const val CAP_PROP_XI_FREE_FFS_SIZE: Int = 581
    const val CAP_PROP_XI_USED_FFS_SIZE: Int = 582
    const val CAP_PROP_XI_FFS_ACCESS_KEY: Int = 583
    const val CAP_PROP_XI_SENSOR_FEATURE_SELECTOR: Int = 585
    const val CAP_PROP_XI_SENSOR_FEATURE_VALUE: Int = 586
    const val CAP_PROP_ARAVIS_AUTOTRIGGER: Int = 600
    const val CAP_PROP_ANDROID_DEVICE_TORCH: Int = 8001
    const val CAP_PROP_IOS_DEVICE_FOCUS: Int = 9001
    const val CAP_PROP_IOS_DEVICE_EXPOSURE: Int = 9002
    const val CAP_PROP_IOS_DEVICE_FLASH: Int = 9003
    const val CAP_PROP_IOS_DEVICE_WHITEBALANCE: Int = 9004
    const val CAP_PROP_IOS_DEVICE_TORCH: Int = 9005
    const val CAP_PROP_GIGA_FRAME_OFFSET_X: Int = 10001
    const val CAP_PROP_GIGA_FRAME_OFFSET_Y: Int = 10002
    const val CAP_PROP_GIGA_FRAME_WIDTH_MAX: Int = 10003
    const val CAP_PROP_GIGA_FRAME_HEIGHT_MAX: Int = 10004
    const val CAP_PROP_GIGA_FRAME_SENS_WIDTH: Int = 10005
    const val CAP_PROP_GIGA_FRAME_SENS_HEIGHT: Int = 10006
    const val CAP_PROP_INTELPERC_PROFILE_COUNT: Int = 11001
    const val CAP_PROP_INTELPERC_PROFILE_IDX: Int = 11002
    const val CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE: Int = 11003
    const val CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE: Int = 11004
    const val CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD: Int = 11005
    const val CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ: Int = 11006
    const val CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT: Int = 11007
    const val CAP_INTELPERC_DEPTH_GENERATOR: Int = 1 shl 29
    const val CAP_INTELPERC_IMAGE_GENERATOR: Int = 1 shl 28
    const val CAP_INTELPERC_IR_GENERATOR: Int = 1 shl 27
    const val CAP_INTELPERC_GENERATORS_MASK: Int =
        CAP_INTELPERC_DEPTH_GENERATOR + CAP_INTELPERC_IMAGE_GENERATOR + CAP_INTELPERC_IR_GENERATOR
    const val CAP_INTELPERC_DEPTH_MAP: Int = 0
    const val CAP_INTELPERC_UVDEPTH_MAP: Int = 1
    const val CAP_INTELPERC_IR_MAP: Int = 2
    const val CAP_INTELPERC_IMAGE: Int = 3
    const val CAP_PROP_GPHOTO2_PREVIEW: Int = 17001
    const val CAP_PROP_GPHOTO2_WIDGET_ENUMERATE: Int = 17002
    const val CAP_PROP_GPHOTO2_RELOAD_CONFIG: Int = 17003
    const val CAP_PROP_GPHOTO2_RELOAD_ON_CHANGE: Int = 17004
    const val CAP_PROP_GPHOTO2_COLLECT_MSGS: Int = 17005
    const val CAP_PROP_GPHOTO2_FLUSH_MSGS: Int = 17006
    const val CAP_PROP_SPEED: Int = 17007
    const val CAP_PROP_APERTURE: Int = 17008
    const val CAP_PROP_EXPOSUREPROGRAM: Int = 17009
    const val CAP_PROP_VIEWFINDER: Int = 17010
    const val CAP_PROP_IMAGES_BASE: Int = 18000
    const val CAP_PROP_IMAGES_LAST: Int = 19000
}
