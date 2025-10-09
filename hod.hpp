#ifndef HOD_HPP
#define HOD_HPP
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_rng.h>

const gsl_rng_type * T;
gsl_rng * rand_ins;

void init_rand_hod(void){
        gsl_rng_env_setup();
        T = gsl_rng_default;
        rand_ins = gsl_rng_alloc (T);
}

void set_seed(unsigned int s){
	gsl_rng_set(rand_ins,s);
}

bool prob_cen(double M,double logMmin,double sigmalogM){
	return ((0.5 * (1.+gsl_sf_erf((log10(M)-logMmin)/sigmalogM))) >= gsl_rng_uniform(rand_ins));
}

void free_rand_hod(void){
        gsl_rng_free (rand_ins);
}

#endif
