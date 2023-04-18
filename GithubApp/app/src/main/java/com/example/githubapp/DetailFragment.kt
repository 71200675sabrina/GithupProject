package com.example.githubapp

import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.githubapp.databinding.FragmentDetailBinding


class DetailFragment : Fragment() {

    private lateinit var binding: FragmentDetailBinding
    private lateinit var detailViewModel: DetailViewModel

    companion object {
        const val ARG_POSITION = "position"
        const val ARG_USERNAME = ""
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = FragmentDetailBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        var position = 0
        var username = arguments?.getString(ARG_USERNAME)

        Log.d("arguments: position", position.toString())
        Log.d("arguments: username", username.toString())

        detailViewModel =
            ViewModelProvider(requireActivity(), ViewModelProvider.NewInstanceFactory()).get(
                DetailViewModel::class.java
            )
        detailViewModel.getUserDetail(query = String())

        arguments?.let {
            position = it.getInt(ARG_POSITION)
            username = it.getString(ARG_USERNAME)
        }

        if (position == 1) {
            username?.let { detailViewModel.getFollowers(it) }
            detailViewModel.followers.observe(viewLifecycleOwner) {
                setUserFollow(it)
            }
        } else {
            username?.let { detailViewModel.getFollowing(it) }
            detailViewModel.following.observe(viewLifecycleOwner) {
                setUserFollow(it)
            }
        }

    }


    private fun setUserFollow(listFollowing: List<ItemsItem>) {
        binding.apply {
            binding.rvFollow.layoutManager = LinearLayoutManager(requireActivity())
            val adapter = UserAdapter(listFollowing)
            binding.rvFollow.adapter = adapter

        }
    }
}
