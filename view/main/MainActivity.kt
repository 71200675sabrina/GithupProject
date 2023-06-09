package com.bangkit.skincareku.view.main

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.MenuItem
import androidx.fragment.app.Fragment
import com.bangkit.skincareku.R
import com.bangkit.skincareku.databinding.ActivityMainBinding
import com.bangkit.skincareku.view.main.buyProduct.BuyProductFragment
import com.bangkit.skincareku.view.main.dashboard.DashboardFragment
import com.bangkit.skincareku.view.main.faceAnalyze.FaceAnalyzeActivity
import com.bangkit.skincareku.view.main.feed.FeedFragment
import com.bangkit.skincareku.view.signup.SignupActivity

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        supportActionBar?.hide()

        val dashboardFragment = DashboardFragment()
        val feedFragment = FeedFragment()
        val buyProductFragment = BuyProductFragment()

        setCurrentFragment(dashboardFragment)

        val btmNav = binding.bottomNavMain

        btmNav.setOnNavigationItemSelectedListener { item ->
            when (item.itemId) {
                R.id.menu_home -> {
                    setCurrentFragment(dashboardFragment)
                    true
                }
                R.id.menu_faceAnalyze -> {
                    val intent = Intent(this, FaceAnalyzeActivity::class.java)
                    startActivity(intent)
                    true
                }
                R.id.menu_feed -> {
                    setCurrentFragment(feedFragment)
                    true
                }
                R.id.menu_buyProduct -> {
                    setCurrentFragment(buyProductFragment)
                    true
                }
                else -> false
            }
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {

        return super.onOptionsItemSelected(item)
    }

    fun setCurrentFragment(fragment: Fragment)=
        supportFragmentManager.beginTransaction().apply {
            replace(R.id.fl_main,fragment)
            commit()
        }
}