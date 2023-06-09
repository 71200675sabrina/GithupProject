package com.bangkit.skincareku.view.signup

import android.content.ContentValues
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.bangkit.skincareku.networking.data.DataManager
import com.bangkit.skincareku.networking.response.RegisterResponse
import com.bangkit.skincareku.networking.retrofit.ApiConfig
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class SignUpViewModel : ViewModel() {

    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading
    private val _isConnectionError = MutableLiveData<Boolean>()
    val isConnectionError: LiveData<Boolean> = _isConnectionError
    private val _isSuccessful = MutableLiveData<Boolean>()
    val isSuccessful: LiveData<Boolean> = _isSuccessful

    fun register(name: String, email: String, password: String) {
        _isConnectionError.value = false
        _isLoading.value = true

        val client = ApiConfig.getApiService().register(name, email, password)
        client.enqueue(object : Callback<RegisterResponse> {
            override fun onResponse(
                call: Call<RegisterResponse>,
                response: Response<RegisterResponse>
            ) {
                _isLoading.value = false

                if(response.isSuccessful) {
                    _isSuccessful.value = true
                } else {
                    _isSuccessful.value = false
                    Log.e(ContentValues.TAG, "onFailure: ${response}")
                }
            }

            override fun onFailure(call: Call<RegisterResponse>, t: Throwable) {
                _isLoading.value = false
                _isConnectionError.value = true
                Log.e(ContentValues.TAG, "onFailure2: ${t.message.toString()}")
            }
        })
    }
}