package com.bangkit.skincareku.view.main.faceAnalyze

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.bangkit.skincareku.databinding.ActivityFaceAnalyzeBinding
import com.bangkit.skincareku.ml.SkincareKu
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class FaceAnalyzeActivity : AppCompatActivity() {
    private lateinit var binding: ActivityFaceAnalyzeBinding
    private lateinit var imageView: ImageView
    private lateinit var button: Button
    private lateinit var outputTextView: TextView
    private var GALLERY_REQUEST_CODE = 123

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityFaceAnalyzeBinding.inflate(layoutInflater)
        setContentView(binding.root)

        imageView = binding.imageView
        button = binding.bntCaptureImage
        outputTextView = binding.outputTextView
        val buttonLoad = binding.btnLoadImage

        button.setOnClickListener {
            if (ContextCompat.checkSelfPermission(
                    this,
                    android.Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
            ){
                takePicturePreview.launch(null)
            } else{
                requestPermission.launch(android.Manifest.permission.CAMERA)
            }
        }
        buttonLoad.setOnClickListener {
            if (ContextCompat.checkSelfPermission(
                    this,
                    android.Manifest.permission.READ_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED
            ){
                val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                intent.type = "image/*"
                val mimeType = arrayOf("image/jpeg", "image/png", "image/jpg")
                intent.putExtra(Intent.EXTRA_MIME_TYPES, mimeType)
                intent.flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
                onResult.launch(intent)
            } else {
                requestPermission.launch(android.Manifest.permission.READ_EXTERNAL_STORAGE)
            }
        }

    }
    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                takePicturePreview.launch(null)
            } else {
                Toast.makeText(this, "Permission Denied! Try Again.", Toast.LENGTH_SHORT).show()
            }
        }

    private val takePicturePreview =
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
            if (bitmap != null) {
                imageView.setImageBitmap(bitmap)
                outputGenerator(bitmap)
            }
        }

    private val onResult =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            Log.i("TAG", "This is the result: ${result.data} ${result.resultCode}")
            onResultRecived(GALLERY_REQUEST_CODE, result)
        }

    private fun onResultRecived(requestCode: Int, result: ActivityResult?) {
        when (requestCode) {
            GALLERY_REQUEST_CODE -> {
                if (result?.resultCode == Activity.RESULT_OK) {
                    result.data?.data?.let { uri ->
                        Log.i("TAG", "onResultRecived: $uri")
                        val bitmap =
                            BitmapFactory.decodeStream(contentResolver.openInputStream(uri))
                        imageView.setImageBitmap(bitmap)
                        outputGenerator(bitmap)
                    }
                } else {
                    Log.e("TAG", "onActivityResult: error in selecting image")
                }
            }
        }
    }

    private fun outputGenerator(bitmap: Bitmap) {
        //declaring tensorflow lite model veriable
        val skinmodel = SkincareKu.newInstance(this)

        // Converting bitmap into tensorflow image
        val newBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val tfimage = TensorImage.fromBitmap(newBitmap)

        val tensorBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 150, 150, 3), DataType.FLOAT32)
        tfimage.load(tensorBuffer)

        // Process the image using trained model and sort it in descending order
        val outputs = skinmodel.process(tensorBuffer)
        val outputTensor = outputs.outputFeature0AsTensorBuffer

        // Getting result having high probability
        val floatArray = outputTensor.floatArray
        var maxIndex = 0
        var maxValue = floatArray[0]

        for (i in 1 until floatArray.size) {
            if (floatArray[i] > maxValue) {
                maxIndex = i
                maxValue = floatArray[i]
            }
        }

        val labels = getLabelForIndex(maxIndex)


        // Setting output text
        outputTextView.text = labels
        Log.i("TAG", "outputGenerator: $labels")


        // Releases model resources if no longer used.
        skinmodel.close()
    }

    private fun getLabelForIndex(index: Int): String {
        // Daftar label yang diketahui
        val labels = listOf("Jerawat", "Clear face", "Komedo")

        // Mengambil label berdasarkan indeks
        return if (index >= 0 && index < labels.size) {
            labels[index]
        } else {
            "Unknown Label"
        }
    }

}