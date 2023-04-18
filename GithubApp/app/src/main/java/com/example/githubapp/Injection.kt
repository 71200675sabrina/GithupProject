package com.example.githubapp

import android.content.Context
import com.example.githubapp.database.UserRoomDatabase
import com.example.githubapp.repository.UserRepository


object Injection {

    fun provideRepository(context: Context): UserRepository{
        val apiService = ApiConfig.getApiService()
        val database = UserRoomDatabase.getInstance(context)
        val dao = database.userDao()
        return UserRepository.getInstance(apiService, dao)
    }
}