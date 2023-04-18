package com.example.githubapp.theme

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider

class SwitchViewModelFactory (private val pref: SettingPreferences) : ViewModelProvider.NewInstanceFactory() {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(SwitchViewModel::class.java)) {
            return SwitchViewModel(pref) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class: " + modelClass.name)
    }
}