package com.bangkit.skincareku.networking.data

import android.content.Context

class DataManager(context: Context) {
    private val preferences = context.getSharedPreferences("skincareku", Context.MODE_PRIVATE)
}