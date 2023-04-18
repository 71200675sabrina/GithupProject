package com.example.githubapp.favorite


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.githubapp.database.FavoriteUser
import com.example.githubapp.repository.UserRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class FavoriteViewModel (private val userRepository: UserRepository) : ViewModel() {
     fun getFavoriteUser() = userRepository.getFavoriteUsers()
     fun AddDelUser(user: FavoriteUser, isfavorite: Boolean) {
         viewModelScope.launch (Dispatchers.IO ) {
             if (isfavorite){
                 userRepository.deleteFavoriteUser(user, false)
             } else {
                 userRepository.addFavoriteUser(user, true)
             }
         }
     }
}