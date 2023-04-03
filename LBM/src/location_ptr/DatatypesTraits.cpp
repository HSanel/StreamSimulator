// ========================================================================= //
//                                                                           //
// Filename: DatatypesTraits.cpp                                             
//                                                                           //
//                                                                           //
// Author: Fraunhofer Institut fuer Graphische Datenverarbeitung (IGD)       //
// Competence Center Interactive Engineering Technologies                    //
// Fraunhoferstr. 5                                                          //
// 64283 Darmstadt, Germany                                                  //
//                                                                           //
// Rights: Copyright (c) 2013 by Fraunhofer IGD.                             //
// All rights reserved.                                                      //
// Fraunhofer IGD provides this product without warranty of any kind         //
// and shall not be liable for any damages caused by the use                 //
// of this product.                                                          //
//                                                                           //
// ========================================================================= //
//                                                                           //
// Creation Date : 11.2012 dweber                                            
//                                                                           //
// ========================================================================= //

#include "DatatypesTraits.h"
#include "Real.h"

#define LOCAL_TEMPLATE_INSTANTIATION_CLASS_FOR_SCALARS(REALTYPE) \
template<> const unsigned int Trait<REALTYPE>::mTypeSize = 1;\
template<> const unsigned int Trait<REALTYPE>::size = 1;\
template<> const REALTYPE Trait<REALTYPE>::zeroElem = static_cast<REALTYPE>(0.0f);\
template<> const REALTYPE Trait<REALTYPE>::identityElem = static_cast<REALTYPE>(1.0f);\
\
template<> const Vector3T<REALTYPE> Trait<Vector3T<REALTYPE>>::zeroElem = Vector3T<REALTYPE>::zero();\
template<> const Vector3T<REALTYPE> Trait<Vector3T<REALTYPE>>::identityElem = Vector3T<REALTYPE>(static_cast<REALTYPE>(1.0f), static_cast<REALTYPE>(1.0f), static_cast<REALTYPE>(1.0f));\
template<> const unsigned int Trait<Vector3T<REALTYPE>>::size = 3;\
\
template<> const unsigned int Trait<Matrix3T<REALTYPE>>::mTypeSize = 3;\
template<> const unsigned int Trait<Matrix3T<REALTYPE>>::size = 9;\
template<> const Matrix3T<REALTYPE> Trait<Matrix3T<REALTYPE>>::zeroElem = Matrix3T<REALTYPE>::zero;\
template<> const Matrix3T<REALTYPE> Trait<Matrix3T<REALTYPE>>::identityElem = Matrix3T<REALTYPE>(Vector3T<REALTYPE>::e1, Vector3T<REALTYPE>::e2, Vector3T<REALTYPE>::e3);
EXECUTE_MACRO_FOR_ALL_SCALARS(LOCAL_TEMPLATE_INSTANTIATION_CLASS_FOR_SCALARS)
#undef LOCAL_TEMPLATE_INSTANTIATION_CLASS_FOR_SCALARS


