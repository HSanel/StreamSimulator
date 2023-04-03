#ifndef LBM_DOMAIN_BOUNDARY
#define LBM_DOMAIN_BOUNDARY

enum BoundSide { right, left, top, bottom, back, front };

//templates
template<typename T, size_t D>
struct VELOCITY_BOUND;

template<typename T, size_t D>
struct PRESSURE_BOUND;

template<typename T, size_t D>
struct VELOCITY_BOUND_DEV;

template<typename T, size_t D>
struct PRESSURE_BOUND_DEV;

//Structs mit Konstruktor erzeugen error: keine dynamischen instanzierung
//specialisation structs for cuda
#pragma region Bounds_2D
template<typename T>
struct VELOCITY_BOUND_DEV<T, 2>
{
	BoundSide side;
	T u_w;		//velocity
};

template<typename T>
struct PRESSURE_BOUND_DEV<T, 2> {
	BoundSide side;
	T dp_w;			//pressure
};


//specialisation
template<typename T>
struct VELOCITY_BOUND<T, 2> {
	BoundSide side;
	T u_w;		//velocity

	VELOCITY_BOUND(BoundSide side, T velocity_u_w) :side(side), u_w(velocity_u_w) {}

	void copyToDev(VELOCITY_BOUND_DEV<T, 2>& boundDev)
	{
		boundDev.side = side;
		boundDev.u_w = u_w;
	}
};

template<typename T>
struct PRESSURE_BOUND<T, 2> {
	BoundSide side;
	T dp_w;			//pressure

	PRESSURE_BOUND(BoundSide side, T pressure_dp_w) :side(side), dp_w(pressure_dp_w) {}

	void copyToDev(PRESSURE_BOUND_DEV<T, 2>& boundDev)
	{
		boundDev.side = side;
		boundDev.dp_w = dp_w;
	}
};

#pragma endregion

template<typename T>
struct VELOCITY_BOUND_DEV<T, 3>
{
	BoundSide side;
	T u_w;		//velocity
};


template<typename T>
struct PRESSURE_BOUND_DEV<T, 3> {
	BoundSide side;
	T dp_w;			//pressure
};



template<typename T>
struct VELOCITY_BOUND<T, 3> {
	BoundSide side;
	T u_w;		//velocity

	VELOCITY_BOUND(BoundSide side, T velocity_u_w) :side(side), u_w(velocity_u_w) {}
	
	void copyToDev(VELOCITY_BOUND_DEV<T, 3>& boundDev)
	{
		boundDev.side = side;
		boundDev.u_w = u_w;
	}
};

template<typename T>
struct PRESSURE_BOUND<T, 3> {
	BoundSide side;
	T dp_w;			//pressure

	PRESSURE_BOUND(BoundSide side, T pressure_dp_w) :side(side), dp_w(pressure_dp_w) {}

	void copyToDev(PRESSURE_BOUND_DEV<T, 3>& boundDev)
	{
		boundDev.side = side;
		boundDev.dp_w = dp_w;
	}
};

#endif