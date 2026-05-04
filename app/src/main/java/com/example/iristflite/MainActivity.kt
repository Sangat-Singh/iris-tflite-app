package com.example.irisftflite

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.provider.MediaStore
import android.speech.tts.TextToSpeech
import android.util.Size
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.ExecutorService

class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {

    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>
    private lateinit var tts: TextToSpeech

    private var lastSpokenLabels = mutableStateOf<Set<String>>(emptySet())
    private var isVoiceOn = mutableStateOf(true)
    private var useBackCamera = mutableStateOf(true)
    private var previewView: PreviewView? = null
    private var cameraControl: CameraControl? = null
    private var currentZoom by mutableStateOf(1f)
    private var lastDetections: List<Detection> = emptyList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestPermission()

        interpreter = Interpreter(loadModel("yolov10.tflite"))
        labels = assets.open("labels.txt").bufferedReader().readLines()
        tts = TextToSpeech(this, this)

        setContent {
            Surface(modifier = Modifier.fillMaxSize()) {
                CameraScreen()
            }
        }
    }

    override fun onDestroy() {
        tts.stop()
        tts.shutdown()
        super.onDestroy()
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
        }
    }

    private fun requestPermission() {
        val launcher = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
            if (!it) finish()
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            launcher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun loadModel(filename: String): ByteBuffer {
        val fileDescriptor = assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    @Composable
    fun CameraScreen() {
        val context = LocalContext.current
        previewView = remember { PreviewView(context) }
        val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
        var detections by remember { mutableStateOf(emptyList<Detection>()) }
        val executor: ExecutorService = remember { Executors.newSingleThreadExecutor() }

        BoxWithConstraints(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
                .pointerInput(Unit) {
                    detectTransformGestures { _, _, zoom, _ ->
                        cameraControl?.setZoomRatio((currentZoom * zoom).coerceIn(1f, 10f))
                        currentZoom = (currentZoom * zoom).coerceIn(1f, 10f)
                    }
                }
        ) {
            val screenWidth = constraints.maxWidth.toFloat()
            val screenHeight = constraints.maxHeight.toFloat()

            AndroidView(factory = { previewView!! }, modifier = Modifier.fillMaxSize()) {
                cameraProviderFuture.get().also { cameraProvider ->
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView!!.surfaceProvider)
                    }

                    val analyzer = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 640)) // deprecated, but safe for now
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    analyzer.setAnalyzer(executor) { imageProxy ->
                        detections = analyzeImage(imageProxy, screenWidth, screenHeight)
                        imageProxy.close()
                    }

                    val selector = if (useBackCamera.value) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA

                    cameraProvider.unbindAll()
                    val camera = cameraProvider.bindToLifecycle(this@MainActivity, selector, preview, analyzer)
                    cameraControl = camera.cameraControl
                }
            }

            BoundingBoxes(detections)

            Column(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(onClick = { isVoiceOn.value = !isVoiceOn.value }) {
                        Text(if (isVoiceOn.value) "Mute Voice" else "Unmute Voice")
                    }
                    Button(onClick = { useBackCamera.value = !useBackCamera.value }) {
                        Text("Switch Camera")
                    }
                }
                Button(onClick = { captureCurrentFrame(context) }) {
                    Text("Capture Image")
                }
                Slider(
                    value = currentZoom,
                    onValueChange = {
                        currentZoom = it
                        cameraControl?.setZoomRatio(it)
                    },
                    valueRange = 1f..10f,
                    steps = 9,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp)
                )
            }
        }
    }

    private fun analyzeImage(imageProxy: ImageProxy, screenWidth: Float, screenHeight: Float): List<Detection> {
        val rotation = imageProxy.imageInfo.rotationDegrees
        val bitmap = imageProxy.toBitmap() ?: return emptyList()
        val rotated = rotateBitmap(bitmap, rotation)
        val inputBitmap = Bitmap.createScaledBitmap(rotated, 640, 640, true)

        val inputBuffer = ByteBuffer.allocateDirect(1 * 640 * 640 * 3 * 4).apply { order(ByteOrder.nativeOrder()) }
        val pixels = IntArray(640 * 640)
        inputBitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)

        for (pixel in pixels) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
            inputBuffer.putFloat((pixel and 0xFF) / 255f)
        }

        val output = Array(1) { Array(300) { FloatArray(6) } }
        interpreter.run(inputBuffer, output)

        val detections = postProcess(output, screenWidth, screenHeight)
        lastDetections = detections
        if (isVoiceOn.value) speakDetections(detections)
        return detections
    }

    private fun postProcess(output: Array<Array<FloatArray>>, screenWidth: Float, screenHeight: Float): List<Detection> {
        val results = mutableListOf<Detection>()
        val isPortrait = screenHeight > screenWidth
        val scaleX = if (isPortrait) screenWidth / 640f else screenHeight / 640f
        val scaleY = if (isPortrait) screenHeight / 640f else screenWidth / 640f

        for (row in output[0]) {
            val left = row[0]
            val top = row[1]
            val right = row[2]
            val bottom = row[3]
            val score = row[4]
            val classIdx = row[5].toInt()

            if (score > 0.5f && classIdx in labels.indices) {
                val label = labels[classIdx]
                val rect = RectF(
                    left * 640f * scaleX,
                    top * 640f * scaleY,
                    right * 640f * scaleX,
                    bottom * 640f * scaleY
                )
                results.add(Detection(label, score, rect))
            }
        }
        return results
    }

    private fun speakDetections(detections: List<Detection>) {
        val newLabels = detections.map { it.label }.toSet()
        val toSpeak = newLabels.subtract(lastSpokenLabels.value)
        if (toSpeak.isNotEmpty()) {
            val sentence = toSpeak.joinToString(", ")
            tts.speak("Detected: $sentence", TextToSpeech.QUEUE_FLUSH, null, null)
            lastSpokenLabels.value = newLabels
        }
    }

    private fun captureCurrentFrame(context: android.content.Context) {
        previewView?.bitmap?.let { bitmap ->
            val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(mutableBitmap)

            val paintBox = Paint().apply {
                color = android.graphics.Color.RED
                style = Paint.Style.STROKE
                strokeWidth = 6f
                isAntiAlias = true
            }

            val paintText = Paint().apply {
                color = android.graphics.Color.RED
                textSize = 48f
                typeface = Typeface.DEFAULT_BOLD
                isAntiAlias = true
            }

            val paintTextBg = Paint().apply {
                color = android.graphics.Color.BLACK
                style = Paint.Style.FILL
            }

            for (det in lastDetections) {
                canvas.drawRect(det.bbox, paintBox)
                val label = "${det.label} ${(det.confidence * 100).toInt()}%"
                val textWidth = paintText.measureText(label)
                val textHeight = paintText.textSize
                canvas.drawRect(
                    det.bbox.left,
                    det.bbox.top - textHeight,
                    det.bbox.left + textWidth,
                    det.bbox.top,
                    paintTextBg
                )
                canvas.drawText(label, det.bbox.left, det.bbox.top - 8f, paintText)
            }

            val filename = "vision_${System.currentTimeMillis()}.jpg"
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/VisionAssist")
            }

            val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            uri?.let {
                context.contentResolver.openOutputStream(it)?.use { outputStream ->
                    mutableBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
                    Toast.makeText(context, "Image saved with boxes!", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }

    private fun rotateBitmap(bitmap: Bitmap, rotation: Int): Bitmap {
        if (rotation == 0) return bitmap
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    @Composable
    fun BoundingBoxes(detections: List<Detection>) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            detections.forEach {
                drawRect(
                    color = Color.Red,
                    topLeft = Offset(it.bbox.left, it.bbox.top),
                    size = androidx.compose.ui.geometry.Size(it.bbox.width(), it.bbox.height()),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 4f)
                )
                val text = "${it.label} ${(it.confidence * 100).toInt()}%"
                val paint = Paint().apply {
                    color = android.graphics.Color.RED
                    textSize = 40f
                    isAntiAlias = true
                    typeface = Typeface.DEFAULT_BOLD
                }
                val textWidth = paint.measureText(text)
                val textHeight = paint.textSize
                drawContext.canvas.nativeCanvas.drawRect(
                    it.bbox.left,
                    it.bbox.top - textHeight,
                    it.bbox.left + textWidth,
                    it.bbox.top,
                    Paint().apply { color = android.graphics.Color.BLACK }
                )
                drawContext.canvas.nativeCanvas.drawText(text, it.bbox.left, it.bbox.top - 6f, paint)
            }
        }
    }

    data class Detection(val label: String, val confidence: Float, val bbox: RectF)
}
