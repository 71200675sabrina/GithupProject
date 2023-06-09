package com.bangkit.skincareku.view.signup

import android.app.ProgressDialog
import android.content.ContentValues.TAG
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import com.bangkit.skincareku.R
import com.bangkit.skincareku.databinding.ActivitySignupBinding
import com.bangkit.skincareku.view.login.LoginActivity
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.ktx.auth
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking

class SignupActivity : AppCompatActivity() {
    private lateinit var binding: ActivitySignupBinding
    private lateinit var signUpViewModel: SignUpViewModel
    private lateinit var progressDialog: ProgressDialog
    private lateinit var progressDialogInformation: ProgressDialog

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySignupBinding.inflate(layoutInflater)
        setContentView(binding.root)

        supportActionBar?.hide()

        signUpViewModel = SignUpViewModel()

        setupViewModel()
        setup()
    }

    private fun setupViewModel() {
        signUpViewModel.isSuccessful.observe(this, {
            if (it == true) {
                val alertDialog = AlertDialog.Builder(this)
                    .setTitle(getString(R.string.success_register_title))
                    .setMessage(getString(R.string.success_register))
                    .setCancelable(false)
                    .setPositiveButton(getString(R.string.login_now)) { _, _ ->
                        val intent = Intent(this, LoginActivity::class.java)
                        intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TASK or Intent.FLAG_ACTIVITY_NEW_TASK
                        startActivity(intent)
                    }
                alertDialog.show()
            }else {
                Toast.makeText(this, getString(R.string.email_taken), Toast.LENGTH_SHORT).show()
            }
        })

        signUpViewModel.isLoading.observe(this, {
            if (it) {
                progressDialog = ProgressDialog(this)
                progressDialog.show()
                progressDialog.setContentView(R.layout.item_progress_dialog)
                progressDialog.window?.setBackgroundDrawableResource(android.R.color.transparent)
            }else {
                progressDialog.dismiss()
            }
        })

        signUpViewModel.isConnectionError.observe(this, {
            if(it){
                AlertDialog.Builder(this).apply {
                    setTitle(getString(R.string.no_internet))
                    setMessage(getString(R.string.connection_error))
                    setPositiveButton(getString(R.string.retry)) { _, _ ->
                        signup()
                    }
                    create()
                    show()
                }
            }
        })
    }

    private fun setup () {
        binding.tvLogin.setOnClickListener {
            startActivity(Intent(this, LoginActivity::class.java))
        }
        binding.btnSignup.setOnClickListener{
            signup()
        }
    }

    private fun signup() {
        val name = binding.signupEtName.text.toString()
        val email = binding.signupEtEmail.text.toString()
        val password = binding.signupEtPassword.text.toString()
        val confirmPassword = binding.signupEtPasswordConfirmation.text.toString()
        when {
            name.isEmpty() -> {
                binding.signupTilName.error = getString(R.string.insert_name)
            }
            email.isEmpty() -> {
                binding.signupTilEmail.error = getString(R.string.insert_email)
            }
            password.isEmpty() -> {
                binding.signupTilPassword.error = getString(R.string.insert_password)
            }
            password.length < 6 -> {
                binding.signupTilPassword.error = getString(R.string.password_length)
            }
            confirmPassword.isEmpty() -> {
                binding.signupTilPassword.error = getString(R.string.insert_confirm_password)
            }
            password != confirmPassword -> {
                binding.signupEtPasswordConfirmation.error = getString(R.string.password_not_match)
            }
            else -> {
                signUpViewModel.register(name, email, password)
            }
        }
    }
}