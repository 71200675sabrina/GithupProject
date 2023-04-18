package com.example.githubapp


import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.githubapp.favorite.FavoriteViewModel
import com.example.githubapp.repository.UserRepository


class ViewModelFactory private constructor(private val userRepository: UserRepository):
        ViewModelProvider.NewInstanceFactory(){
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(FavoriteViewModel::class.java)){
            return FavoriteViewModel(userRepository) as T
        }
        throw java.lang.IllegalArgumentException("Unknown ViewModel Class: ${modelClass.name}")
    }

    companion object{
        @Volatile
        private var instance: ViewModelFactory? = null
        fun getInstance(context: Context): ViewModelFactory =
            instance ?: synchronized(this) {
                instance ?: ViewModelFactory(Injection.provideRepository(context))
            }.also { instance = it }

    }
}

