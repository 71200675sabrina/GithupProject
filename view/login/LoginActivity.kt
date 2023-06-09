package com.bangkit.skincareku.view.login

import android.app.ProgressDialog
import android.content.ContentValues.TAG
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import com.bangkit.skincareku.R
import com.bangkit.skincareku.databinding.ActivityLoginBinding
import com.bangkit.skincareku.networking.data.DataManager
import com.bangkit.skincareku.view.main.MainActivity
import com.bangkit.skincareku.view.signup.SignupActivity
import com.google.firebase.FirebaseApp
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.ktx.auth
import com.google.firebase.ktx.Firebase

class LoginActivity : AppCompatActivity() {
    private lateinit var binding: ActivityLoginBinding
    private lateinit var auth: FirebaseAuth
    private lateinit var loginViewModel: LoginViewModel
    private lateinit var progressDialog: ProgressDialog

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLoginBinding.inflate(layoutInflater)
        setContentView(binding.root)

        supportActionBar?.hide()
        FirebaseApp.initializeApp(this)
        auth = FirebaseAuth.getInstance()

        setupViewModel()
        setup()

    }

    public override fun onStart() {
        super.onStart()
        // Check if user is signed in (non-null) and update UI accordingly.
        val currentUser = auth.currentUser
        if (currentUser != null) {
//            val intent = Intent(this, MainActivity::class.java)
//            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TASK or Intent.FLAG_ACTIVITY_NEW_TASK
//            startActivity(intent)
        }
    }

    private fun setupViewModel() {
        val dataManager = DataManager(this)
        loginViewModel = LoginViewModel(dataManager)

        loginViewModel.isSuccessful.observe(this, {
            if (it == true) {
                val intent = Intent(this, MainActivity::class.java)
                intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TASK or Intent.FLAG_ACTIVITY_NEW_TASK
                startActivity(intent)
            }else {
                Toast.makeText(this, getString(R.string.login_error), Toast.LENGTH_SHORT).show()
            }
        })

        loginViewModel.isLoading.observe(this, {
            if (it) {
                progressDialog = ProgressDialog(this)
                progressDialog.show()
                progressDialog.setContentView(R.layout.item_progress_dialog)
                progressDialog.window?.setBackgroundDrawableResource(android.R.color.transparent)
            }else {
                progressDialog.dismiss()
            }
        })

        loginViewModel.isConnectionError.observe(this, {
            if(it){
                AlertDialog.Builder(this).apply {
                    setTitle(getString(R.string.no_internet))
                    setMessage(getString(R.string.connection_error))
                    setPositiveButton(getString(R.string.retry)) { _, _ ->
                        login()
                    }
                    create()
                    show()
                }
            }

        })
    }

    private fun setup () {
        binding.tvSignup.setOnClickListener {
            startActivity(Intent(this, SignupActivity::class.java))
        }

        binding.btnLogin.setOnClickListener {
            login()
        }
    }

    private fun login() {
        val email = binding.loginEtEmail.text.toString()
        val password = binding.loginEtPassword.text.toString()
        when {
            email.isEmpty() -> {
                binding.loginTilEmail.error = getString(R.string.insert_email)
            }
            !email.matches(Regex("^[A-Za-z](.*)([@]{1})(.{1,})(\\.)(.{1,})")) -> {
                binding.loginTilEmail.error = getString(R.string.invalid_email)
            }
            password.isEmpty() -> {
                binding.loginTilPassword.error = getString(R.string.insert_password)
            }
            else -> {
                Log.d("login_data", "$email - $password")
                loginViewModel.login(email, password)
            }
        }

    }
}