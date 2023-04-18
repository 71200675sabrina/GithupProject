package com.example.githubapp


import android.app.SearchManager
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SearchView
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.githubapp.databinding.ActivityMainBinding
import com.example.githubapp.favorite.FavoriteActivity
import com.example.githubapp.theme.SwitchMainActivity


class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var mainViewModel: MainViewModel


    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        supportActionBar?.show()

        mainViewModel = ViewModelProvider(this).get(MainViewModel::class.java)
        val mainViewModel = ViewModelProvider(this, ViewModelProvider.NewInstanceFactory()).get(MainViewModel::class.java)
        mainViewModel.listUser.observe(this, {
            ItemsItem -> setListData(ItemsItem)
        })

        mainViewModel.isLoading.observe(this, {
            showLoading(it)
        })

        val layoutManager = LinearLayoutManager(this)
        binding.rvUser.layoutManager = layoutManager
        val itemDecoration = DividerItemDecoration(this, layoutManager.orientation)
        binding.rvUser.addItemDecoration(itemDecoration)


    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        val inflater = menuInflater
        inflater.inflate(R.menu.option_menu, menu)

        val searchManager = getSystemService(Context.SEARCH_SERVICE) as SearchManager
        val searchView = menu?.findItem(R.id.search)?.actionView as SearchView

        searchView.setSearchableInfo(searchManager.getSearchableInfo(componentName))
        searchView.queryHint = resources.getString(R.string.search_hint)
        searchView.setOnQueryTextListener(object : SearchView.OnQueryTextListener{
            override fun onQueryTextSubmit(query: String?): Boolean {
                if (query != null) {
                    mainViewModel.findUser(query)
                }
                searchView.clearFocus()
                return true
            }

            override fun onQueryTextChange(newText: String?): Boolean {
                return false
            }
        })
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when(item.itemId){
            R.id.it_favorite -> {val intent = Intent(this, FavoriteActivity::class.java)
            startActivity(intent)
            }
            R.id.it_dark_mode ->{val intent = Intent(this, SwitchMainActivity::class.java)
            startActivity(intent)}
        }
        return super.onOptionsItemSelected(item)
    }


    private fun setListData(listUser : List<ItemsItem>){
        val adapter = UserAdapter(listUser)
        binding.rvUser.adapter = adapter
    }

    private fun showLoading(isLoading: Boolean){
        if (isLoading) {
            binding.loadRv.visibility = View.VISIBLE
        } else {
            binding.loadRv.visibility = View.GONE
        }
    }


}