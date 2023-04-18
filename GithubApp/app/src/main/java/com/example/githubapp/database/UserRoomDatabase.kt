package com.example.githubapp.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [FavoriteUser::class], version = 1)
abstract class UserRoomDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao

    companion object{
        @Volatile
        private var INSTANCE: UserRoomDatabase? = null
        fun getInstance (context: Context): UserRoomDatabase = INSTANCE?: synchronized(this){
            INSTANCE ?: Room.databaseBuilder(
                context.applicationContext,
                UserRoomDatabase::class.java, "favorite_database"
            ).build()
        }
    }
}