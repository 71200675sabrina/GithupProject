package com.example.githubapp

import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.viewModels
import androidx.annotation.StringRes
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.viewpager2.widget.ViewPager2
import com.bumptech.glide.Glide
import com.example.githubapp.database.FavoriteUser
import com.example.githubapp.databinding.UserDetailBinding
import com.example.githubapp.favorite.FavoriteViewModel
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator

class DetailUserActivity : AppCompatActivity() {

    private lateinit var binding: UserDetailBinding
    private lateinit var viewModel: DetailViewModel
    private val favoriteViewModel by viewModels<FavoriteViewModel> (){ ViewModelFactory.getInstance(application) }

    companion object{
        const val USER_KEY = "key_data"

        @StringRes
        private val TAB_TITLES = intArrayOf(
            R.string.tab_text_1,
            R.string.tab_text_2
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = UserDetailBinding.inflate(layoutInflater)
        setContentView(binding.root)


        val username = intent.getStringExtra(USER_KEY)
        val bundle = Bundle()
        bundle.putString(USER_KEY, username)

        viewModel = ViewModelProvider(this, ViewModelProvider.NewInstanceFactory()).get(DetailViewModel::class.java)

        val loginUser = intent.getStringExtra(USER_KEY)
        binding.tvName.text = loginUser

        val sectionsPagerAdapter = SectionsPagerAdapter(this)
        sectionsPagerAdapter.username = loginUser.toString()

        val viewPager: ViewPager2 = findViewById(R.id.view_pager)
        viewPager.adapter = sectionsPagerAdapter

        val tabs: TabLayout = findViewById(R.id.tab_detail)
        TabLayoutMediator(tabs, viewPager) {tabs, position ->

            tabs.text = resources.getString(TAB_TITLES[position])
        }.attach()

        supportActionBar?.elevation = 0f

        viewModel.detailUser.observe(this){
                detailUser -> setDetailUser(detailUser)
                showLoading(false)
        }

        if (username != null){
            showLoading(true)
            viewModel.getUserDetail(username)
        }

        viewModel.isLoading.observe(this){
            showLoading(it)
        }


    }

    private fun setIconFavorite(isfavorite: Boolean){
        binding.fbFavorite?.apply {
            if (isfavorite) {
                setImageDrawable(ContextCompat.getDrawable(this@DetailUserActivity, R.drawable.full_favorite))
            }else{
                setImageDrawable(ContextCompat.getDrawable(this@DetailUserActivity, R.drawable.favorite_border))
            }
        }
    }

    private fun setDetailUser(Data: DetailUserResponse) {
        Glide.with(this@DetailUserActivity)
            .load(Data.avatarUrl)
            .into(binding.ivUserdetail)
        binding.tvName.text = Data.login
        binding.tvUsername.text = Data.name
        binding.tvFollowers.text = Data.followers.toString()
        binding.tvFollowing.text = Data.following.toString()

        favoriteViewModel.getFavoriteUser().observe(this) {favoriteList ->
            val isfavorite = favoriteList.any { it.username == Data.name }

            setIconFavorite(isfavorite)

            binding.fbFavorite?.setOnClickListener{
                val favoriteCondition = Data.name?.let { FavoriteUser(it, Data.name,
                    avatarUrl = Data.avatarUrl,
                    followingUrl = Data.followingUrl,
                    followersUrl = Data.followersUrl, false) }

                try {
                    if (favoriteCondition != null ) favoriteViewModel.AddDelUser(favoriteCondition,favoriteList.any { it.id == Data.name })
                } catch (e: Exception){
                    Toast.makeText(this, e.toString(), Toast.LENGTH_SHORT).show()
                }

                if (isfavorite){
                    Toast.makeText(this, "User di hapus dari favorite", Toast.LENGTH_SHORT).show()
                    setIconFavorite(isfavorite)
                }else{
                    Toast.makeText(this, "User ditambahkan ke favorite", Toast.LENGTH_SHORT).show()
                    setIconFavorite(isfavorite)
                }
            }

        }
    }

    private fun showLoading(isLoading: Boolean){
        if (isLoading) {
            binding.progressBar.visibility = View.VISIBLE
        } else {
            binding.progressBar.visibility = View.GONE
        }
    }


}


