package com.example.githubapp.favorite

import android.os.Bundle
import android.os.PersistableBundle
import android.view.View
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.githubapp.ViewModelFactory
import com.example.githubapp.databinding.ActivityFavoriteBinding


class FavoriteActivity : AppCompatActivity() {
    private lateinit var binding: ActivityFavoriteBinding
    private lateinit var adapter: FavoriteUserAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityFavoriteBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val factory: ViewModelFactory = ViewModelFactory.getInstance(application)
        val favoriteViewModel: FavoriteViewModel by viewModels { factory }

        adapter = FavoriteUserAdapter()

        binding.apply {
            rvFavorite.layoutManager = LinearLayoutManager(this@FavoriteActivity)
            rvFavorite.setHasFixedSize(true)
            rvFavorite.adapter = adapter
        }

        favoriteViewModel.getFavoriteUser().observe(this){favoriteList ->
            adapter.updateList(favoriteList)
            binding.rvLoad.visibility = View.GONE
        }

    }
}