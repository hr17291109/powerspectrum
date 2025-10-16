#pragma once

#include <cstdlib>
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include "fileloader.hpp"


// flat LCDM cosmology used for the conversion from redshifts to the comoving distances
const double Omegam_dummy = 0.31;

// THIS WORKS ONLY FOR Q0000!!
double Hz_Q0000(double redshift, double Omegam0)
{
	double H0(100.);
	double Omegal0(1.-Omegam0);
	double time = 1.0/(1.0+redshift);
	double w = -1.36524163118195533752;
	return H0*sqrt(Omegam0/(time*time*time)+Omegal0*pow(1/time, 3*(1+w)));
}

double Hz_LCDM(double redshift, double Omegam0) // Valid for flat-Lambda CDM cosmologies
{
	double H0(100.);
	double Omegal0(1.-Omegam0);
	double time = 1.0/(1.0+redshift);
	return H0*sqrt(Omegam0/(time*time*time)+Omegal0);
}

double Hz(double redshift, std::string fname){ // This is to read from Hubble table and return 1/(aH)

	// std::cout << "Checking the expansion table in " << fname << std::endl;

	double H0(100.); // In [h km/s/Mpc]
	double scalefac(1.0/(1.0+redshift));

	gsl_spline *spl;
	gsl_interp_accel *acc;

	std::vector<double> atable;
	std::vector<double> Etable; // H/H0

	std::string tmp, token;
    std::istringstream stream;
    std::ifstream fin;
    int nlines, ncolumns;
    std::string::size_type comment_start = 0;
    get_file_format(fname, nlines, ncolumns);

	// std::cout << "Found " << ncolumns << " entries, " << nlines << " lines." << std::endl;

	atable.resize(nlines);
	Etable.resize(nlines);
    nlines = 0;

	double tmp1, tmp2;
    fin.open(fname.c_str());
	while(!fin.eof()){
		std::getline(fin, tmp);
		if ((comment_start = tmp.find("#")) != std::string::size_type(-1))
            tmp = tmp.substr(0, comment_start);
		std::istringstream stream(tmp);
		int nelem = 0;
        while (stream >> token)
            nelem++;
		if (nelem == ncolumns) {
            std::istringstream stream_read(tmp);
            nelem = 0;
			while (stream_read >> token) {
				nelem++;

				if(nelem==1) atable[nlines] = atof(token.c_str());
				if(nelem==2) Etable[nlines] = atof(token.c_str());
			}
			nlines ++;
		}
	}
	fin.close();
	// std::cout << nlines << " lines read successfully." << std::endl;

	acc = gsl_interp_accel_alloc();
	spl = gsl_spline_alloc(gsl_interp_cspline,atable.size());
	gsl_spline_init(spl,&atable[0],&Etable[0],atable.size());
	double Ez(gsl_spline_eval(spl,scalefac,acc)); // Ez = H/H0

	gsl_spline_free(spl);
	gsl_interp_accel_free(acc);

	return H0 * Ez;
  // return sqrt(time)
  //   /(100.0*time
  //     *sqrt(Omegam0/time/time/time
  //           +(1.0-Omegam0-Omegal0)/time/time
  //           +Omegal0));
}


double vel_to_disp(double redshift, std::string fname){ // no gadget factor sqrt(a)
	double time = 1.0/(1.0+redshift);
	//double Omegal0(1.-Omegam0);
	//return sqrt(time)/(100.0*time*sqrt(Omegam0/time/time/time+(1.0-Omegam0-Omegal0)/time/time+Omegal0));
	//return 1./(100.0*time*sqrt(Omegam0/time/time/time+Omegal0));
	return 1./(time*Hz(redshift,fname));
}

double vel_to_disp_LCDM(double redshift, double Omegam0){ // no gadget factor sqrt(a)
	double time = 1.0/(1.0+redshift);
	double Omegal0(1.-Omegam0);
	//return sqrt(time)/(100.0*time*sqrt(Omegam0/time/time/time+(1.0-Omegam0-Omegal0)/time/time+Omegal0));
	return 1./(100.0*time*sqrt(Omegam0/time/time/time+Omegal0));
	// return 1./(time*Hz(redshift,fname));
}

double inv_Hz(double redshift, void * params){
	std::string fname = *(std::string *) params;
	return 1./Hz(redshift,fname);
}

double inv_Hz_LCDM(double redshift, void * params){
	double Omegam0 = *(double *) params;
	return 1./Hz_LCDM(redshift,Omegam0);
}

double inv_Hz_Q0000(double redshift, void * params){
	double Omegam0 = *(double *) params;
	return 1./Hz_Q0000(redshift,Omegam0);
}

double ZtoComovingD(double redshift, std::string fname){
	double cc(2.99792458E5);

	double result, error;
	gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
	gsl_function F;
	F.function = &inv_Hz;
	F.params = &fname;

	gsl_integration_qags (&F, 0, redshift, 0, 1e-7, 1000, w, &result, &error);

	gsl_integration_workspace_free (w);
	return cc*result;
}

double ZtoComovingD_LCDM(double redshift, double Omegam0){
	double cc(2.99792458E5);

	double result, error;
	gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
	gsl_function F;
	F.function = &inv_Hz_LCDM;
	F.params = &Omegam0;

	gsl_integration_qags (&F, 0, redshift, 0, 1e-7, 1000, w, &result, &error);

	gsl_integration_workspace_free (w);
	return cc*result;
}

double ZtoComovingD_Q0000(double redshift, double Omegam0){
	double cc(2.99792458E5);

	double result, error;
	gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
	gsl_function F;
	F.function = &inv_Hz_Q0000;
	F.params = &Omegam0;

	gsl_integration_qags (&F, 0, redshift, 0, 1e-7, 1000, w, &result, &error);

	gsl_integration_workspace_free (w);
	return cc*result;
}


double OmegamZ_LCDM(double redshift, double Omegam0){
	double H0(100.);
	return Omegam0 * (1.+redshift)*(1.+redshift)*(1.+redshift) * H0*H0 / (Hz_LCDM(redshift,Omegam0)*Hz_LCDM(redshift,Omegam0));
}

// double delta_vir(double redshift, std::string fname){
// 	double x(OmegamZ(redshift,fname)-1.);
// 	return 18. * M_PI * M_PI + 82. * x - 39. * x * x;
// }

// double Mvir_to_Rvir(double Mvir, double redshift, std::string fname){ // Rvir in Mpc/h, comoving
// 	double H0(100.);
// 	double rhoc0(2.77536627E11); // [M_sun/h] / [Mpc/h]^3
// 	double rhocZ = rhoc0 * (Hz(redshift,Omegam0)*Hz(redshift,Omegam0))/(H0*H0);
// 	double del_vir = delta_vir(redshift,Omegam0);
// 	return pow(3.*Mvir / (4*M_PI * del_vir * rhocZ),1./3.)*(1.+redshift);
// }

// double M200_to_R200(double M200, double Omegam0){ // Rvir in Mpc/h, comoving
// 	double rhoc0(2.77536627E11); // [M_sun/h] / [Mpc/h]^3
// 	return pow(3.*M200 / (4*M_PI * 200 * rhoc0 * Omegam0),1./3.); 
// }

// double get_sfac_gadget(double redshift, double Omegam0, double Omegal0){ // dark energy w not yet implemented !!!
//   double time = 1.0/(1.0+redshift);

//   return sqrt(time)
//     /(100.0*time
//       *sqrt(Omegam0/time/time/time
//             +(1.0-Omegam0-Omegal0)/time/time
//             +Omegal0));
// }

double get_sfac(double redshift, std::string fname){ // This is to read from Hubble table and return 1/(aH)

	// std::cout << "Checking the expansion table in " << fname << std::endl;

	double H0(100.); // In [h km/s/Mpc]
	double scalefac(1.0/(1.0+redshift));

	gsl_spline *spl;
	gsl_interp_accel *acc;

	std::vector<double> atable;
	std::vector<double> Etable; // H/H0

	std::string tmp, token;
    std::istringstream stream;
    std::ifstream fin;
    int nlines, ncolumns;
    std::string::size_type comment_start = 0;
    get_file_format(fname, nlines, ncolumns);

	// std::cout << "Found " << ncolumns << " entries, " << nlines << " lines." << std::endl;

	atable.resize(nlines);
	Etable.resize(nlines);
    nlines = 0;

	double tmp1, tmp2;
    fin.open(fname.c_str());
	while(!fin.eof()){
		std::getline(fin, tmp);
		if ((comment_start = tmp.find("#")) != std::string::size_type(-1))
            tmp = tmp.substr(0, comment_start);
		std::istringstream stream(tmp);
		int nelem = 0;
        while (stream >> token)
            nelem++;
		if (nelem == ncolumns) {
            std::istringstream stream_read(tmp);
            nelem = 0;
			while (stream_read >> token) {
				nelem++;

				if(nelem==1) atable[nlines] = atof(token.c_str());
				if(nelem==2) Etable[nlines] = atof(token.c_str());
			}
			nlines ++;
		}
	}
	fin.close();
	// std::cout << nlines << " lines read successfully." << std::endl;

	acc = gsl_interp_accel_alloc();
	spl = gsl_spline_alloc(gsl_interp_cspline,atable.size());
	gsl_spline_init(spl,&atable[0],&Etable[0],atable.size());
	double Ez(gsl_spline_eval(spl,scalefac,acc));

	gsl_spline_free(spl);
	gsl_interp_accel_free(acc);

	double sfac(1./(scalefac * H0 * Ez));
	std::cout << "km/s to Mpc/h: " << sfac << std::endl;

	return sfac;
  // return sqrt(time)
  //   /(100.0*time
  //     *sqrt(Omegam0/time/time/time
  //           +(1.0-Omegam0-Omegal0)/time/time
  //           +Omegal0));
}
