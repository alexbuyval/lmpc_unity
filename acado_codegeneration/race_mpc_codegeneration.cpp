/*
 *    This file is part of ACADO Toolkit.
 *
 *    ACADO Toolkit -- A Toolkit for Automatic Control and Dynamic Optimization.
 *    Copyright (C) 2008-2014 by Boris Houska, Hans Joachim Ferreau,
 *    Milan Vukov, Rien Quirynen, KU Leuven.
 *    Developed within the Optimization in Engineering Center (OPTEC)
 *    under supervision of Moritz Diehl. All rights reserved.
 *
 *    ACADO Toolkit is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with ACADO Toolkit; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include <acado_code_generation.hpp>

USING_NAMESPACE_ACADO

#define SEPARATE_BRAKE

IntermediateState LateralForce(IntermediateState alpha, IntermediateState Fz, IntermediateState muy, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7)
{


    Fz = Fz/1000;           //Conversion [N] - [kN]

    double C = a0;                 //Shape factor
    IntermediateState muy0 = a1 * Fz + a2;      // Lateral friction coefficient nominal [-]
    muy = muy * 1000;         // Lateral friction coefficient operacional
    IntermediateState D = muy0 * Fz;            // muy = lateral friction coefficient
    IntermediateState BCD = a3 * sin(2 * atan(Fz/a4)); // Cornering stiffness
    IntermediateState E = a6 * Fz + a7;         // Curvature factor
    IntermediateState B = BCD/(C * D);          // stiffness factor
    IntermediateState ALPHAeq = muy0/muy*(180 /  M_PI * alpha);   // Equivalent slip angle
    // Reference characteristics
    IntermediateState fy = D * sin(C * atan(B * ALPHAeq - E*(B * ALPHAeq - atan(B * ALPHAeq))));
    // Lateral force
    return -muy/muy0*(fy);
}


int main( void )
{
    //
    // OCP parameters
    //
    // Step length
    double Ts = 2.0;
    // Number of shootin' intervals
    int N = 25;
    // Number of integrator steps per shootin' interval
    int Ni = 4;

    // INTRODUCE THE VARIABLES:
    // -------------------------
    DifferentialState eY;
    DifferentialState ePsi;
    DifferentialState vx;
    DifferentialState vy;
    DifferentialState dPSI;
    DifferentialState t;
    DifferentialState STEER;
    IntermediateState s;
    IntermediateState Fl, FlFL, FlFR, FlRL, FlRR;
    IntermediateState ALPHAF, ALPHAR;

	    
    Control TRACTION; //Traction Force [0..1] Если используется только один TRACTION, то значение в kN
    Control dSTEER;
    #ifdef SEPARATE_BRAKE
        Control BRAKE; //Brake force [0..1]
    #endif

    OnlineData cu;
    IntermediateState ro = 1.0/cu;

    //Static parameters.
    IntermediateState muy = 1.0; //muy - friction coefficient
    double I = 1260;
    double g=9.81;
    double m=1270;
    double wb = 2.708; //wheelbase
    double a = 1.510;  //mR/m*wb;
    double b=wb-a;
    double mF = m*(1-a/wb);
    double mR = m*(a/wb);
    double nF = 2;
    double nR = 2;
    double max_brake_force = 10000.0;
    double max_traction_force = 5000.0;
    double c = 0.85;

    double cr2 = 1.4; //0.0041*m;
    double cr0 = 200; //N //0.2315*m;

    //double ro = -100; //radius of curvature
    //double cu = 1/ro;  //curvature

    // DEFINE THE DYNAMIC SYSTEM:
    // --------------------------
    DifferentialEquation f;

    //Slip angles
    IntermediateState y1 = vy + a * dPSI;
    ALPHAF = -2*atan(y1/(sqrt(vx*vx+ y1*y1) + vx)) + STEER;
    IntermediateState y2 = b * dPSI - vy;
    ALPHAR = 2*atan(y2/(sqrt(vx*vx+ y2*y2) + vx));

    IntermediateState FzFL = mF*g/2; // Vertical load @ F [N]
    IntermediateState FzFR = mF*g/2;
    IntermediateState FzRL = mR*g/2;
    IntermediateState FzRR = mR*g/2;

    //increase vertical forces proportional by long velocity
    FzFL = FzFL+80*vx;
    FzFR = FzFR+80*vx;
    FzRL = FzRL+80*vx;
    FzRR = FzRR+80*vx;

    //transfer vertical force on out wheels proportional by angular velocity
    IntermediateState Ft = (FzFL+FzFR)*dPSI/2;
    FzFL = FzFL-Ft;
    FzFR = FzFR+Ft;
    IntermediateState Rt = (FzRL+FzRR)*dPSI/2;
    FzRL = FzRL-Rt;
    FzRR = FzRR+Rt;

#ifdef SEPARATE_BRAKE
    Fl =  TRACTION*max_traction_force-cr2*vx*vx-cr0 - BRAKE*max_brake_force;
#else
    Fl =  TRACTION*1000-cr2*vx*vx-cr0;
#endif

    FlFL = Fl/4;
    FlFR = Fl/4;
    FlRL = Fl/4;
    FlRR = Fl/4;

#ifdef SEPARATE_BRAKE
    muy=muy-1.0*BRAKE;
#else
    //muy=muy-sqrt(TRACTION*TRACTION); //надо как-то нормализовать в диапазон [0..1]
#endif

    IntermediateState FcFL = -LateralForce(ALPHAF, FzFL, muy,1.0, 0, 800, 5000, 50, 0, 0, -1);
    IntermediateState FcFR = -LateralForce(ALPHAF, FzFR, muy,1.0, 0, 800, 5000, 50, 0, 0, -1);
    IntermediateState FcRL = -LateralForce(ALPHAR, FzRL, muy,1.0, 0, 800, 5000, 50, 0, 0, -1);
    IntermediateState FcRR = -LateralForce(ALPHAR, FzRR, muy,1.0, 0, 800, 5000, 50, 0, 0, -1);

    IntermediateState cosS = cos(STEER);
    IntermediateState sinS = sin(STEER);

    IntermediateState FxFL = FlFL*cosS - FcFL*sinS;
    IntermediateState FxFR = FlFR*cosS - FcFR*sinS;
    IntermediateState FxRL = FlRL;
    IntermediateState FxRR = FlRR;

    IntermediateState FyFL = FlFL*sinS + FcFL*cosS;
    IntermediateState FyFR = FlFR*sinS + FcFR*cosS;
    IntermediateState FyRL = FcRL;
    IntermediateState FyRR = FcRR;

    IntermediateState cosePsi = cos(ePsi);
    IntermediateState sinePsi = sin(ePsi);

    s = (ro/(ro-eY)*(vx*cosePsi-vy*sinePsi));

    f << dot(eY) ==  (vx * sinePsi + vy * cosePsi)/s;
    f << dot(ePsi) == dPSI/s - cu;
    f << dot(vx) == (FxFL+FxFR+FxRL+FxRR + m * vy * dPSI) / m /s;
    f << dot(vy) == (FyFL+FyFR+FyRL+FyRR - m * vx * dPSI) / m /s;
    f << dot(dPSI) == (a*(FyFL+FyFR)-b*(FyRL+FyRR)+c*(FxFR-FxFL+FxRR-FxRL)) / I / s;
    f << dot(t) == 1/s;
    f << dot(STEER) == dSTEER;


    //
    // MHE PROBLEM FORMULATION
    //
    OCP ocp(0.0, N * Ts, N);

    ocp.subjectTo( f );
#ifdef SEPARATE_BRAKE
    ocp.subjectTo( 0.0 <= TRACTION <= 1.0 );
#else
    ocp.subjectTo( -max_brake_force <= TRACTION <= max_traction_force );
#endif
    ocp.subjectTo( -0.2 <= dSTEER <= 0.2 );
    ocp.subjectTo( -4.5 <= dSTEER*vx <= 4.5 );
#ifdef SEPARATE_BRAKE
    ocp.subjectTo( 0.0 <= BRAKE <= 1.0 );
#endif
    ocp.subjectTo( -10.0 <= eY <= 10.0 );
    ocp.subjectTo( -1.0 <= ePsi <= 1.0 );
    ocp.subjectTo( 2.0 <= vx <= 30.0 );
    ocp.subjectTo( -10.0 <= vy <= 10.0 );
    ocp.subjectTo( -2.5 <= dPSI <= 2.5 );
    ocp.subjectTo( 0.0 <= t <= 50.0 );
    ocp.subjectTo( -0.3 <= STEER <= 0.3 );

    // DEFINE LEAST SQUARE FUNCTION:
    // -----------------------------
    Function h;

    h << eY;
    h << ePsi;
    h << dPSI;
    h << TRACTION;
    h << dSTEER;
#ifdef SEPARATE_BRAKE
    h << BRAKE;
#endif
    h << STEER;
    h << vy;

    // Weighting matrices and measurement functions
    // certain values are defined online 
    BMatrix W = eye<bool>( h.getDim() );

    Function hN;
    hN << t;

    BMatrix WN = eye<bool>( hN.getDim() );

	
    ocp.minimizeLSQ(W, h);
    ocp.minimizeLSQEndTerm(WN, hN);
    ocp.setNOD(1);

    OCPexport mpc( ocp );

    mpc.set(INTEGRATOR_TYPE, INT_IRK_GL4);
    //mhe.set(ABSOLUTE_TOLERANCE, 1.0e-1);
    mpc.set(NUM_INTEGRATOR_STEPS, N * Ni);
    mpc.set(IMPLICIT_INTEGRATOR_NUM_ITS, 1);

    mpc.set(HESSIAN_APPROXIMATION, GAUSS_NEWTON);
    mpc.set(DISCRETIZATION_TYPE, MULTIPLE_SHOOTING);
    mpc.set( GENERATE_TEST_FILE, NO );
    mpc.set(CG_HARDCODE_CONSTRAINT_VALUES, NO);
    mpc.set(HOTSTART_QP, YES);
    mpc.set(PRINTLEVEL, HIGH);
    mpc.set(CG_USE_VARIABLE_WEIGHTING_MATRIX, NO);
    mpc.set(QP_SOLVER, QP_HPMPC);
    mpc.set(SPARSE_QP_SOLUTION, SPARSE_SOLVER);
    //mpc.set(QP_SOLVER, QP_QPOASES);
    //mpc.set(SPARSE_QP_SOLUTION, CONDENSING);
    // NOTE: This is crucial for export of MHE!
    mpc.set(FIX_INITIAL_STATE, YES);

    //mhe.set( LEVENBERG_MARQUARDT, 1e-10 );

    if (mpc.exportCode("roborace_mpc_export") != SUCCESSFUL_RETURN)
		exit( EXIT_FAILURE );

    mpc.printDimensionsQP( );

	return EXIT_SUCCESS;
}
