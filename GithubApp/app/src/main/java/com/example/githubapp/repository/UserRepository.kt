package com.example.githubapp.repository

import androidx.lifecycle.LiveData
import com.example.githubapp.ApiService
import com.example.githubapp.database.FavoriteUser
import com.example.githubapp.database.UserDao

class UserRepository(
    private val apiService : ApiService,
    private val userDao: UserDao){

    fun getFavoriteUsers(): LiveData<List<FavoriteUser>> {
        return userDao.getFavoriteUser()
    }

    fun addFavoriteUser(user: FavoriteUser, conditionFav: Boolean){
        user.isfavorite = conditionFav
        userDao.insertFavorite(user)
    }

    fun deleteFavoriteUser(user: FavoriteUser, conditionFav: Boolean){
        user.isfavorite = conditionFav
        userDao.deleteFavorite(user)
    }

    companion object{
        @Volatile
        private var instance: UserRepository? = null
        fun getInstance(
            apiService: ApiService,
            userDao: UserDao
        ): UserRepository = instance?: synchronized(this){
            instance?: UserRepository(apiService, userDao)
        }.also { instance = it }
    }
}
