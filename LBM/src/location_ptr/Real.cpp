#include "Real.h"

#ifdef WITH_ADOL_C
float unsafe_Convert_ToFloat(adouble const & in)
{
	return static_cast<float>(in.getValue());
}
float unsafe_Convert_ToDouble(adouble const & in)
{
	return in.getValue();
}
#endif

#ifdef WITH_CPPAD
float unsafe_Convert_ToFloat(CppAD::AD<double> const & in)
{
	return static_cast<float>(CppAD::Value(in));
	//std::cout << "CppAD is shitty and does not allow converting while recording. No Gravity? Good. Hope for the best.\n";
	//return 0.0f; // unfortunatly CppAD is shitty. You cannot access Values while recording.
}
double unsafe_Convert_ToDouble(CppAD::AD<double> const & in)
{
	return CppAD::Value(in);
	//std::cout << "CppAD is shitty and does not allow converting while recording. No Gravity? Good. Hope for the best.\n";
	//return 0.0f; // unfortunatly CppAD is shitty. You cannot access Values while recording.
}
std::vector<double> unsafe_Convert_ToDoubleVec(std::vector<CppAD::AD<double>> const & in)
{
	std::vector<double> res(in.size());
	for (int i = 0; i < in.size();i++) {
		res[i] = unsafe_Convert_ToDouble(in[i]);
	}

	return res;
	//std::cout << "CppAD is shitty and does not allow converting while recording. No Gravity? Good. Hope for the best.\n";
	//return 0.0f; // unfortunatly CppAD is shitty. You cannot access Values while recording.
}
#endif