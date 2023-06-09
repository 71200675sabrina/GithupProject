package com.bangkit.skincareku.networking.response

import com.google.gson.annotations.SerializedName

data class RegisterResponse(

	@field:SerializedName("data")
	val data: RegisterData? = null,

	@field:SerializedName("status")
	val status: String? = null,

	@field:SerializedName("message")
	val message: String? = null
)

data class RegisterData(

	@field:SerializedName("user")
	val user: RegisterUser? = null,

	@field:SerializedName("token")
	val token: String? = null
)

data class RegisterProviderDataItem(

	@field:SerializedName("uid")
	val uid: String? = null,

	@field:SerializedName("displayName")
	val displayName: String? = null,

	@field:SerializedName("providerId")
	val providerId: String? = null,

	@field:SerializedName("email")
	val email: String? = null
)

data class RegisterMetadata(

	@field:SerializedName("lastSignInTime")
	val lastSignInTime: Any? = null,

	@field:SerializedName("creationTime")
	val creationTime: String? = null,

	@field:SerializedName("lastRefreshTime")
	val lastRefreshTime: Any? = null
)

data class RegisterUser(

	@field:SerializedName("uid")
	val uid: String? = null,

	@field:SerializedName("emailVerified")
	val emailVerified: Boolean? = null,

	@field:SerializedName("metadata")
	val metadata: RegisterMetadata? = null,

	@field:SerializedName("providerData")
	val providerData: List<RegisterProviderDataItem?>? = null,

	@field:SerializedName("displayName")
	val displayName: String? = null,

	@field:SerializedName("disabled")
	val disabled: Boolean? = null,

	@field:SerializedName("tokensValidAfterTime")
	val tokensValidAfterTime: String? = null,

	@field:SerializedName("email")
	val email: String? = null
)
