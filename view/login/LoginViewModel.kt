package com.bangkit.skincareku.view.login

import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.util.Log
import android.widget.Toast
import androidx.core.content.ContextCompat.startActivity
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.bangkit.skincareku.networking.data.DataManager
import com.bangkit.skincareku.view.main.MainActivity
import com.google.firebase.auth.FirebaseAuth

class LoginViewModel(private val dataManager: DataManager) : ViewModel() {
    private val auth: FirebaseAuth = FirebaseAuth.getInstance()

    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading
    private val _isConnectionError = MutableLiveData<Boolean>()
    val isConnectionError: LiveData<Boolean> = _isConnectionError
    private val _isSuccessful = MutableLiveData<Boolean>()
    val isSuccessful: LiveData<Boolean> = _isSuccessful

    fun login(email: String, password: String) {
        _isConnectionError.value = false
        _isLoading.value = true
        auth.signInWithEmailAndPassword(email, password)
            .addOnCompleteListener(Activity()) { task ->
                if(task.isSuccessful) {
                    Log.d(ContentValues.TAG, "signInWithEmail:success")
                    _isSuccessful.value = true
                    _isLoading.value = false
                    val user = auth.currentUser
                } else {
                    _isSuccessful.value = false
                    _isLoading.value = false
                    Log.w(ContentValues.TAG, "signInWithEmail:failure", task.exception)
                    Log.d("login_error", task.exception?.message.toString())
                    val message = task.exception?.message.toString()
                    if(message.contains("deleted", false) || message.contains("invalid", false)) {
                        _isSuccessful.value = false
                    } else {
                        _isConnectionError.value = true
                    }
                }
            }
    }
}