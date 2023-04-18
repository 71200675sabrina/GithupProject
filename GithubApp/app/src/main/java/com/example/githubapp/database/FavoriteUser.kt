package com.example.githubapp.database

import android.os.Parcelable
import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey
import kotlinx.parcelize.Parcelize

@Entity(tableName = "favorite_user")
@Parcelize
data class FavoriteUser (
    @ColumnInfo(name = "id")
    @PrimaryKey
    var id: String,

    @ColumnInfo(name = "username")
    var username: String,

    @ColumnInfo(name = "avatarUrl")
    var avatarUrl: String,

    @ColumnInfo(name = "followingUrl")
    var followingUrl: String,

    @ColumnInfo(name = "followersUrl")
    var followersUrl: String,

    @ColumnInfo(name = "isfavorite")
    var isfavorite: Boolean
): Parcelable