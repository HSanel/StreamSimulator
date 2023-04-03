// ========================================================================= //
//                                                                           //
// Filename: Vector3.cpp                                                     
//                                                                           //
//                                                                           //
// Author: Fraunhofer Institut fuer Graphische Datenverarbeitung (IGD)       //
// Competence Center Interactive Engineering Technologies                    //
// Fraunhoferstr. 5                                                          //
// 64283 Darmstadt, Germany                                                  //
//                                                                           //
// Rights: Copyright (c) 2012 by Fraunhofer IGD.                             //
// All rights reserved.                                                      //
// Fraunhofer IGD provides this product without warranty of any kind         //
// and shall not be liable for any damages caused by the use                 //
// of this product.                                                          //
//                                                                           //
// ========================================================================= //
//                                                                           //
// Creation Date : 09.2012 Daniel Weber                                      
//                                                                           //
// ========================================================================= //

#include "Vector3.h"

//#include "Matrix3.h"

#include <cmath>
#include <iomanip>

template<typename real>
Vector3T<real> Vector3T<real>::e1(1.0, 0.0, 0.0);
template<typename real>
Vector3T<real> Vector3T<real>::e2(0.0, 1.0, 0.0);
template<typename real>
Vector3T<real> Vector3T<real>::e3(0.0, 0.0, 1.0);

template<typename real>
Vector3T<real>::Vector3T(real *adress)
{
	values[0] = adress[0];
	values[1] = adress[1];
	values[2] = adress[2];
}

template<typename real>
inline std::string Vector3T<real>::str() const
{
	char buffer[64];
	sprintf(buffer, "(%f %f %f)", unsafe_Convert_ToFloat(values[0]), unsafe_Convert_ToFloat(values[1]), unsafe_Convert_ToFloat(values[2]));
	return std::string(buffer);
}

template<typename real>
real Vector3T<real>::maxAbsElement() const
{
	return Trait<real>::max(Trait<real>::abs(values[0]), Trait<real>::max(Trait<real>::abs(values[1]), Trait<real>::abs(values[2])));
}

TEMPLATE_INSTANTIATE_FOR_SCALARS(class Vector3T)
