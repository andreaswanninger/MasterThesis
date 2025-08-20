//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Mohammad R. Hashemi
//

#include "droplet_dynamics_application_variables.h"

namespace Kratos
{
    // External interfacial force, e.g. for including the electromagentic coupling
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(EXT_INT_FORCE)

    //Auxiliary variable to store maximum element size (h_{max})
    KRATOS_CREATE_VARIABLE(double, NODAL_H_MAX)

    // Smoothing surface auxiliary distance
    KRATOS_CREATE_VARIABLE( double, DISTANCE_AUX)
    KRATOS_CREATE_VARIABLE( double, DISTANCE_AUX2)
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(DISTANCE_GRADIENT_AUX)

    // Parallel levelset distance calculator needs an AREA_VARIABLE_AUX
    KRATOS_CREATE_VARIABLE( double, AREA_VARIABLE_AUX)

    // A variable to check if node is on cut element (maybe in a layer farther for future!)
    //KRATOS_CREATE_VARIABLE( double, IS_NEAR_CUT)

    // Contact line calculation
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(NORMAL_VECTOR)
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(TANGENT_VECTOR)
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(CONTACT_VECTOR)
    KRATOS_CREATE_VARIABLE( double, CONTACT_ANGLE)
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(CONTACT_VECTOR_MICRO)
    KRATOS_CREATE_VARIABLE( double, CONTACT_ANGLE_MICRO)
    KRATOS_CREATE_VARIABLE( double, CONTACT_VELOCITY)

    //////////////////////
    // AW 14.5: this code block added
     KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS( VELOCITY_X_GRADIENT)
     KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS( VELOCITY_Y_GRADIENT)
     KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS( VELOCITY_Z_GRADIENT)
     KRATOS_CREATE_VARIABLE( double, DISTANCE_GRADIENT_DIVERGENCE)
     KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS( DISTANCE_GRADIENT_SIZE_GRADIENT)
     KRATOS_CREATE_VARIABLE( double, INTERSECTION_LENGTH_2D)
     KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(ELEMENT_CUT_NORMAL)
     KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(ELEMENT_CUT_NORMAL_AVERAGED)
    // AW 14.5: end of added code block
    ///////////////////////

    // Enriched pressure is an array of NumNodes components defined for elements. Access it using Element.GetValue()
    KRATOS_CREATE_VARIABLE( double, ENRICHED_PRESSURE_1)
    KRATOS_CREATE_VARIABLE( double, ENRICHED_PRESSURE_2)
    KRATOS_CREATE_VARIABLE( double, ENRICHED_PRESSURE_3)
    KRATOS_CREATE_VARIABLE( double, ENRICHED_PRESSURE_4)

    // Last known velocity and pressure to recalculate the last increment
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS( VELOCITY_STAR)
    KRATOS_CREATE_VARIABLE( double, PRESSURE_STAR)

    // Pressure gradient to calculate its jump over interface
    // KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS( PRESSURE_GRADIENT_AUX)

    // Level-set convective velocity
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(CONVECTIVE_VELOCITY)

    // AW 21.4
    KRATOS_CREATE_VARIABLE(bool, quasi_static_contact_angle)
    KRATOS_CREATE_VARIABLE(double, theta_equilibrium_hydrophilic)
    KRATOS_CREATE_VARIABLE(double, theta_equilibrium_hydrophobic)
    KRATOS_CREATE_VARIABLE(double, penalty_coefficient)
    // AW 2.6
    KRATOS_CREATE_VARIABLE(double, X_threshold)
    // AW 25.6
    KRATOS_CREATE_VARIABLE(bool, smooth_scaling)
    KRATOS_CREATE_VARIABLE(double, smooth_scaling_lower_threshold)
    KRATOS_CREATE_VARIABLE(double, smooth_scaling_upper_threshold)

    // AW 28.4: Fitted curvature and normal at the node 
    KRATOS_CREATE_VARIABLE(double, FITTED_CURVATURE)
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(FITTED_NORMAL)

    // AW 2.5: Fitted curvature and normal at the Gauss Point
    KRATOS_CREATE_VARIABLE(double, CURVATURE_FITTED_GAUSS1)
    KRATOS_CREATE_VARIABLE(double, CURVATURE_FITTED_GAUSS2)
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(NORMAL_FITTED_GAUSS1)
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(NORMAL_FITTED_GAUSS2)

    // AW 19.5: Added variables regarding fitting choices
    KRATOS_CREATE_VARIABLE(std::string, FittingType)                     
    KRATOS_CREATE_VARIABLE(bool, UsePartialFitting)                      
    KRATOS_CREATE_VARIABLE(Vector, FittingElementIds) 
    KRATOS_CREATE_VARIABLE(int, NormalEvaluationMode)                   

}
