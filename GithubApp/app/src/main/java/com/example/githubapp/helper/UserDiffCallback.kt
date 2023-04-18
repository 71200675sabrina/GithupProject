package com.example.githubapp.helper

import androidx.recyclerview.widget.DiffUtil
import com.example.githubapp.database.FavoriteUser

class UserDiffCallback(
    private val oldList:List<FavoriteUser>,
    private val newList: List<FavoriteUser>): DiffUtil.Callback() {

    override fun areContentsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
        val old = oldList[oldItemPosition].username
        val latest = newList[newItemPosition].username
        return old == latest
    }

    override fun areItemsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean =
        oldList == newList

    override fun getNewListSize(): Int = newList.size
    override fun getOldListSize(): Int = oldList.size
}