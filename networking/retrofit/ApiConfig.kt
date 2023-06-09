package com.bangkit.skincareku.networking.retrofit

import android.content.Context
import android.util.Log
import com.bangkit.skincareku.networking.data.DataManager
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

class ApiConfig {
    companion object {
        private lateinit var dataManager : DataManager

        fun init(context: Context) {
            dataManager = DataManager(context)
        }


        fun getApiService(): ApiService {

            val client = OkHttpClient.Builder()
                .addInterceptor{chain ->
                    val original = chain.request()
                    val requestBuilder = original.newBuilder()
                        .header("Accept", "application/json")
                    val request = requestBuilder.build()
                    chain.proceed(request)}
                .build()
            val retrofit = Retrofit.Builder()
                .baseUrl("https://services-skincareku-5ctldki4wq-et.a.run.app/")
                .addConverterFactory(GsonConverterFactory.create())
                .client(client)
                .build()
            return retrofit.create(ApiService::class.java)
        }
    }
}