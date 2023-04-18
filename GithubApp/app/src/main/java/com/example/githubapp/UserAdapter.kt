package com.example.githubapp

import android.content.Intent
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import android.widget.ImageView

class UserAdapter(private val listUser: List<ItemsItem>) : RecyclerView.Adapter<UserAdapter.ViewHolder>() {

    override fun onCreateViewHolder(viewGroup: ViewGroup, viewType: Int) =
        ViewHolder(LayoutInflater.from(viewGroup.context).inflate(R.layout.item_user, viewGroup, false))



    override fun onBindViewHolder(viewHolder: ViewHolder, position: Int) {
        val Users = listUser[position]
        viewHolder.tvItem.text = Users.login
        Glide.with(viewHolder.itemView.context)
            .load(Users.avatarUrl)
            .into(viewHolder.imgphoto)

        viewHolder.itemView.setOnClickListener {
            val intentDetail = Intent(viewHolder.itemView.context,DetailUserActivity::class.java)
            intentDetail.putExtra("key_data",Users.login)
            viewHolder.itemView.context.startActivity(intentDetail)
        }

    }

    override fun getItemCount() = listUser.size


    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val tvItem: TextView = view.findViewById(R.id.tv_user)
        val imgphoto : ImageView = view.findViewById(R.id.iv_user)
    }
}


