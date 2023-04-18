package com.example.githubapp.favorite

import android.content.Intent
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.example.githubapp.DetailUserActivity
import com.example.githubapp.database.FavoriteUser
import com.example.githubapp.databinding.ItemUserBinding
import com.example.githubapp.helper.UserDiffCallback

class FavoriteUserAdapter : RecyclerView.Adapter<FavoriteUserAdapter.ViewHolder>(){
    private var listUser = emptyList<FavoriteUser>()

    fun updateList(newList: List<FavoriteUser>){
        val diffCall = DiffUtil.calculateDiff(UserDiffCallback(listUser, newList))
        this.listUser = newList

        diffCall.dispatchUpdatesTo(this)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemUserBinding.inflate(LayoutInflater.from(parent.context),parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(listUser[position])
    }

    override fun getItemCount(): Int = listUser.size

    class ViewHolder(private val binding: ItemUserBinding):RecyclerView.ViewHolder(binding.root){
        fun bind(user: FavoriteUser){
            with(binding){
                Glide.with(itemView.context)
                    .load(user.avatarUrl)
                    .into(ivUser)
                tvUser.text=user.id
            }

            itemView.setOnClickListener{
                val intent = Intent(itemView.context, DetailUserActivity::class.java)
                intent.putExtra(DetailUserActivity.USER_KEY,user.id)
                itemView.context.startActivity(intent)
            }
        }
    }


}