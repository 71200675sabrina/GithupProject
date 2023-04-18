package com.example.githubapp.database

import androidx.lifecycle.LiveData
import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface UserDao {
    @Query("SELECT * FROM favorite_user WHERE isfavorite = 1")
    fun getFavoriteUser() : LiveData<List<FavoriteUser>>

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    fun insertFavorite(user: FavoriteUser)

    @Delete
    fun deleteFavorite(user: FavoriteUser)

}