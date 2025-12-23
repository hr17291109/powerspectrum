#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <fftw3.h>
#include <omp.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_legendre.h>
#include "cosmo.hpp"
#include "select_halos.hpp"
#include "binneddata.hpp"
#include "binneddata_2D.hpp"
#include "hod.hpp"
#include "transfer.hpp"

float THwindow(float kR);
float Gwindow(float kR);
float Sharpk(float kR);

class FieldData{
private:
	bool inFourierSpace;
	size_t nx, ny, nz;
	float Lx, Ly, Lz;
	int ngal;
	std::vector<float> data;
	fftwf_plan forward_plan, backward_plan;
	// static fftwf_plan forward_plan, backward_plan;
	// static bool fft_init;
	void initialize(int a, int b, int c){
		data.resize(((long long int)a)*b*(c/2+1)*2);
		// if(!fft_init){
		// forward_plan = fftwf_plan_dft_r2c_3d(a,b,c,data.data(),(fftwf_complex *)data.data(),FFTW_MEASURE);
		// backward_plan = fftwf_plan_dft_c2r_3d(a,b,c,(fftwf_complex *)data.data(),data.data(),FFTW_MEASURE);
		forward_plan = fftwf_plan_dft_r2c_3d(a,b,c,data.data(),(fftwf_complex *)data.data(),FFTW_ESTIMATE);
		backward_plan = fftwf_plan_dft_c2r_3d(a,b,c,(fftwf_complex *)data.data(),data.data(),FFTW_ESTIMATE);
			// fft_init = true;
		// }
		// forward_plan = fftwf_plan_dft_r2c_3d(a,b,c,data.data(),(fftwf_complex *)data.data(),FFTW_MEASURE);
		// backward_plan = fftwf_plan_dft_c2r_3d(a,b,c,(fftwf_complex *)data.data(),data.data(),FFTW_MEASURE);
	}
	public :
	FieldData() {}
	FieldData(int a, int b, int c, float d, float e, float f, bool g=false) : nx(a), ny(b), nz(c), Lx(d), Ly(e), Lz(f), inFourierSpace(g) {initialize(a,b,c); }
	FieldData(int a, float b, bool c=false) : nx(a), ny(a), nz(a), Lx(b), Ly(b), Lz(b), inFourierSpace(c) {initialize(a,a,a); }
	~FieldData() {if(forward_plan!=NULL) fftwf_destroy_plan(forward_plan); if(backward_plan!=NULL) fftwf_destroy_plan(backward_plan);}
	void fill_data(float);
	void change_space(bool a) {inFourierSpace = a;}
	void do_fft();
	void do_ifft();
	void put_ngal(int a) {ngal = a;}
	int get_nx() const {return nx;}
	int get_ny() const {return ny;}
	int get_nz() const {return nz;}
	float get_Lx() const {return Lx;}
	float get_Ly() const {return Ly;}
	float get_Lz() const {return Lz;}
	int get_ngal() const {return ngal;}
	bool get_space() const {return inFourierSpace;}
	void multiply_data_all(float);
	void subtract_data_all(float);
	FieldData multiply2fields(const FieldData&, const FieldData&, bool);
	void multiply2fields_dummy(const FieldData&, const FieldData&, bool);
	float get_data(long long int n) const {return data[n];}
	float get_data(int,int,int) const;
	float get_data_re(int,int,int) const;
	float get_data_im(int,int,int) const;
	// fftwf_complex get_data_comp(int,int,int) const;
	void add_data(int,int,int,float);
	void put_data(int,int,int,float);
	void put_data_re(int,int,int,float);
	void put_data_im(int,int,int,float);
	void square();
	void clear_elements();
	void assignment(std::vector<myhosthalo_str> &,double,bool,bool);
	void assignment_nonorm(std::vector<myhosthalo_str> &,double,bool,bool);
	void assignment_nonorm(std::vector<myhosthalo_str> &,double,double,bool,bool);
	void assignment(std::vector<myhosthalo_str> &,bool,bool);
	void assignment(std::vector<myhosthalo_str> &,bool,bool,double,int); // for redshift space
	void average2fields(FieldData&, FieldData&);
	void adjust_grid(void);
	void apply_filter(float,std::string);
	void multiply_sqrtPk(std::string,double,double,double,double);
	void multiply_transfer(TransferFunction&,double);
	void divide_transfer(TransferFunction&,double);
	void load(std::string);
	void save_slice(std::string,int);
	BinnedData calc_power_nowindow(int,float,float,bool,int) const;
	BinnedData calc_power_noshot(int,float,float,bool,int) const;
	BinnedData calc_power(int,float,float,bool,int) const;
	BinnedData calc_power(int,float,float,bool,int,std::string,float,float,int) const;
	void calc_power(BinnedData&,int,std::string,float,float,int) const;
	// BinnedData calc_power(int,float,float,bool,int,float,float,float) const;
	// BinnedData calc_power_x(int,float,float,bool,int,float,float,float) const;
	// BinnedData calc_power_y(int,float,float,bool,int,float,float,float) const;
	static BinnedData calc_cross_power_noshot(const FieldData &, const FieldData &, int, float,float,bool,bool,int);
	static BinnedData calc_cross_power_noshot(const FieldData &, const FieldData &, int, float,float,bool,bool,bool,int);
	static BinnedData calc_cross_power(const FieldData &, const FieldData &, int, float,float,bool,bool,int);
	static BinnedData calc_propagator(const FieldData &, const FieldData &, int,float,float,bool,bool);
	BinnedData_2D calc_power_2D(int,float,float,bool,int,float,float,float) const;
	BinnedData calc_correlation(int,float,float,bool) const;
	static BinnedData calc_cross_correlation(FieldData &,FieldData &,int,float,float,bool,bool);
	BinnedData calc_cross_correlation_dummy(const FieldData &,const FieldData &,int,float,float,bool,bool);
	static BinnedData calc_correlation(FieldData &,int,float,float,bool);
	// static void delete_plans(void){if(forward_plan!=NULL) fftwf_destroy_plan(forward_plan); if(backward_plan!=NULL) fftwf_destroy_plan(backward_plan);}
	void zero_pad(int,int,int,int,float);
	// copy constructor
	FieldData(const FieldData& a){
		nx = a.get_nx();
		ny = a.get_ny();
		nz = a.get_nz();
		Lx = a.get_Lx();
		Ly = a.get_Ly();
		Lz = a.get_Lz();
		inFourierSpace = a.get_space();
		initialize(nx,ny,nz);
		#pragma omp parallel for
		for(long long int n=0;n<2*(long long int)nx*(long long int)ny*((long long int)nz/2+1);n++) this->data[n]=a.get_data(n);
	}
	// assignment operator
	FieldData operator=(const FieldData& a){
		nx = a.get_nx();
		ny = a.get_ny();
		nz = a.get_nz();
		Lx = a.get_Lx();
		Ly = a.get_Ly();
		Lz = a.get_Lz();
		inFourierSpace = a.get_space();
		initialize(nx,ny,nz);
		#pragma omp parallel for
		for(long long int n=0;n<2*(long long int)nx*(long long int)ny*((long long int)nz/2+1);n++) this->data[n]=a.get_data(n);
		return(*this);
	}
	void copy_field(const FieldData& a){
		#pragma omp parallel for
		for(long long int n=0;n<2*(long long int)nx*(long long int)ny*((long long int)nz/2+1);n++) data[n]=a.get_data(n);
	}

	FieldData operator*(const FieldData& f){
		#pragma omp parallel for
		for(long long int a=0;a<(long long int)nx;a++)
		for(long long int b=0;b<(long long int)ny;b++)
		for(long long int c=0;c<(long long int)nz/2+1;c++){
			long long int bin = (a*(long long int)ny+b)*(2*((long long int)nz/2+1))+2*c;
			this->data[bin] = data[bin]*f.get_data(bin)+data[bin+1]*f.get_data(bin+1);
			this->data[bin+1] = 0.;
		}
		this->data[0] = 0.;
		return(*this);
	}
};

void FieldData::fill_data(float a){
	#pragma omp parallel for
	for(long long int n=0;n<2*nx*ny*(nz/2+1);n++) data[n] = a;
}

void FieldData::do_fft(){
	std::cerr << "Forward FFT ... ";
	fftwf_execute(forward_plan);
	std::cerr << " done." << std::endl;
	inFourierSpace = true;
	float ng3_inv = 1./(float)(nx*ny*nz);
	#pragma omp parallel for
	for(long long int n=0;n<2*nx*ny*(nz/2+1);n++) data[n] *= ng3_inv;
	std::cerr << "Normalization done." << std::endl;
}

void FieldData::do_ifft(){
	std::cerr << "Backward FFT ... ";
	fftwf_execute(backward_plan);
	std::cerr << " done." << std::endl;
	inFourierSpace = false;
}

void FieldData::multiply_data_all(float a){
	#pragma omp parallel for
	for(long long int n=0;n<2*nx*ny*(nz/2+1);n++) data[n] *= a;
}

void FieldData::subtract_data_all(float a){
	#pragma omp parallel for
	for(long long int n=0;n<2*(long long int)nx*(long long int)ny*((long long int)nz/2+1);n++) data[n] -= a;
}

FieldData FieldData::multiply2fields(const FieldData& f1, const FieldData &f2,bool noDC=false){
	nx = f1.get_nx();
	ny = f1.get_ny();
	nz = f1.get_nz();
	Lx = f1.get_Lx();
	Ly = f1.get_Ly();
	Lz = f1.get_Lz();
	inFourierSpace = f1.get_space();
	initialize(nx,ny,nz);
	#pragma omp parallel for
	for(long long int a=0;a<(long long int)nx;a++)
	for(long long int b=0;b<(long long int)ny;b++)
	for(long long int c=0;c<(long long int)nz/2+1;c++){
		long long int bin = (a*(long long int)ny+b)*(2*((long long int)nz/2+1))+2*c;
		data[bin] = f1.get_data(bin)*f2.get_data(bin)+f1.get_data(bin+1)*f2.get_data(bin+1);
		data[bin+1] = 0.;
	}
	if(noDC) data[0] = 0.;
	return(*this);
}

void FieldData::multiply2fields_dummy(const FieldData& f1, const FieldData &f2,bool noDC=false){
	#pragma omp parallel for
	for(long long int a=0;a<(long long int)nx;a++)
	for(long long int b=0;b<(long long int)ny;b++)
	for(long long int c=0;c<(long long int)nz/2+1;c++){
		long long int bin = (a*(long long int)ny+b)*(2*((long long int)nz/2+1))+2*c;
		data[bin] = f1.get_data(bin)*f2.get_data(bin)+f1.get_data(bin+1)*f2.get_data(bin+1);
		data[bin+1] = 0.;
	}
	if(noDC) data[0] = 0.;
}


// functions for data access: they are not often refered.
float FieldData::get_data (int a, int b, int c) const {
	if(inFourierSpace){
		std::cerr << "This array is in Fourier space. Use get_data_re or get_data_im instead." << std::endl;
		exit(-1);
	}else if( a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz){
		std::cerr << "(" << a << ", " << b << ", " << c << ") is out of the range for (nx,ny,nz)=(" << nx << "," << ny << "," << nz << ")" << std::endl;
		exit(-1);
	}else{
		return data[((long long int)a*(long long int)ny+(long long int)b)*(2*((long long int)nz/2+1))+(long long int)c];
	}
}

float FieldData::get_data_re (int a, int b, int c) const {
	if(!inFourierSpace){
		std::cerr << "This array is in real space. Use get_data instead." << std::endl;
		exit(-1);
	}else if( a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz/2+1 ){
		std::cerr << "(" << a << ", " << b << ", " << c << ") is out of the range for (nx,ny,nz)=(" << nx << "," << ny << "," << nz << ")" << std::endl;
		exit(-1);
	}else{
		return data[((long long int)a*(long long int)ny+(long long int)b)*(2*((long long int)nz/2+1))+(2*(long long int)c)];
	}
}

float FieldData::get_data_im (int a, int b, int c) const {
	if(!inFourierSpace){
		std::cerr << "This array is in real space. Use get_data instead." << std::endl;
		exit(-1);
	}else if( a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz/2+1 ){
		std::cerr << "(" << a << ", " << b << ", " << c << ") is out of the range for (nx,ny,nz)=(" << nx << "," << ny << "," << nz << ")" << std::endl;
		exit(-1);
	}else{
		return data[((long long int)a*(long long int)ny+(long long int)b)*(2*((long long int)nz/2+1))+(2*(long long int)c+1)];
	}
}

// fftwf_complex FieldData::get_data_comp (int a, int b, int c) const {
// 	if(!inFourierSpace){
// 		std::cerr << "This array is in real space. Use get_data instead." << std::endl;
// 		exit(-1);
// 	}else if( a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz/2+1 ){
// 		std::cerr << "(" << a << ", " << b << ", " << c << ") is out of the range for (nx,ny,nz)=(" << nx << "," << ny << "," << nz << ")" << std::endl;
// 		exit(-1);
// 	}else{
// 		return ((fftwf_complex *)data.data())[(a*ny+b)*(nz/2+1)+c];
// 	}
// }

void FieldData::add_data (int a, int b, int c, float d){
	if(inFourierSpace){
		std::cerr << "The function add_data does not work in Fourier space." << std::endl;
		exit(-1);
	}else{
		#pragma omp atomic
		data[((long long int)a*(long long int)ny+(long long int)b)*(2*((long long int)nz/2+1))+(long long int)c] += d;
	}
}

void FieldData::put_data (int a, int b, int c, float d){
	if(inFourierSpace){
		std::cerr << "This array is in Fourier space. Use put_data_re or put_data_im instead." << std::endl;
		exit(-1);
	}else{
		data[((long long int)a*(long long int)ny+(long long int)b)*(2*((long long int)nz/2+1))+(long long int)c] = d;
	}
}

void FieldData::put_data_re (int a, int b, int c, float d){
	if(!inFourierSpace){
		std::cerr << "This array is in real space. Use put_data instead." << std::endl;
		exit(-1);
	}else{
		data[(a*ny+b)*(2*(nz/2+1))+(2*c)] = d;
	}
}

void FieldData::put_data_im (int a, int b, int c, float d){
	if(!inFourierSpace){
		std::cerr << "This array is in real space. Use put_data instead." << std::endl;
		exit(-1);
	}else{
		data[(a*ny+b)*(2*(nz/2+1))+(2*c+1)] = d;
	}
}

void FieldData::square (){
	#pragma omp parallel for
	for(long long int a=0;a<(int)nx;a++)
	for(long long int b=0;b<ny;b++)
	for(long long int c=0;c<nz/2+1;c++){
		data[(a*(long long int)ny+b)*(2*((long long int)nz/2+1))+(2*c)] = data[(a*(long long int)ny+b)*(2*((long long int)nz/2+1))+(2*c)]*data[(a*(long long int)ny+b)*(2*((long long int)nz/2+1))+(2*c)]+data[(a*(long long int)ny+b)*(2*((long long int)nz/2+1))+(2*c+1)]*data[(a*(long long int)ny+b)*(2*((long long int)nz/2+1))+(2*c+1)];
		data[(a*(long long int)ny+b)*(2*((long long int)nz/2+1))+(2*c+1)] = 0.;
	}

}

void FieldData::clear_elements (){
	#pragma omp parallel for
	for(long long int a=0;a<2*(long long int)nx*(long long int)ny*((long long int)nz/2+1);a++)
	data[a] = 0;
}

void FieldData::assignment(std::vector<myhosthalo_str> &D,double nh,bool CIC,bool interlace){


	int ntake((int)(nh*Lx*Ly*Lz));

	if(CIC){
		std::cerr << "start CIC density assignment of " << ntake << " points to ("
		<< nx << "," << ny << "," << nz << ") grid points" << std::endl;
	}else{
		std::cerr << "start NGP density assignment of " << ntake << " points to ("
		<< nx << "," << ny << "," << nz << ") grid points" << std::endl;
	}
	double P_tot(0);

	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		int i1, i2, i3;
		int i1p, i2p, i3p;
		float u, v, w;
		#pragma omp for schedule(guided)
		for(int n=0;n<ntake;n++){

			if(CIC){
				u = (nx*D[n].pos[0])/Lx;
				v = (ny*D[n].pos[1])/Ly;
				w = (nz*D[n].pos[2])/Lz;
			}else{ // FOR NGP
				u = (nx*D[n].pos[0])/Lx+0.5;
				v = (ny*D[n].pos[1])/Ly+0.5;
				w = (nz*D[n].pos[2])/Lz+0.5;
			}
			if(interlace){
				u += 0.5;
				v += 0.5;
				w += 0.5;
			}
			i1 = floor(u);
			i2 = floor(v);
			i3 = floor(w);

			if(interlace){
				i1 -= 1;
				i2 -= 1;
				i3 -= 1;
			}

			while(i1 >= (int)nx) i1 -= (int)nx;
			while(i2 >= (int)ny) i2 -= (int)ny;
			while(i3 >= (int)nz) i3 -= (int)nz;
			while(i1 < 0) i1 += (int)nx;
			while(i2 < 0) i2 += (int)ny;
			while(i3 < 0) i3 += (int)nz;

			if(!CIC){
				this->add_data( i1,  i2,  i3, 1.);
			}else{
				u -= i1;
				v -= i2;
				w -= i3;

				i1p = i1 + 1;
				i2p = i2 + 1;
				i3p = i3 + 1;

				if(i1p >= (int)nx) i1p -= (int)nx;
				if(i2p >= (int)ny) i2p -= (int)ny;
				if(i3p >= (int)nz) i3p -= (int)nz;

				this->add_data( i1,  i2,  i3, (1.-u)*(1.-v)*(1.-w));
				this->add_data( i1,  i2, i3p, (1.-u)*(1.-v)*(w)   );
				this->add_data( i1, i2p,  i3, (1.-u)*(v)   *(1.-w));
				this->add_data( i1, i2p, i3p, (1.-u)*(v)   *(w)   );
				this->add_data(i1p,  i2,  i3, (u)   *(1.-v)*(1.-w));
				this->add_data(i1p,  i2, i3p, (u)   *(1.-v)*(w)   );
				this->add_data(i1p, i2p,  i3, (u)   *(v)   *(1.-w));
				this->add_data(i1p, i2p, i3p, (u)   *(v)   *(w)   );
			}
			#pragma omp atomic
			P_tot += 1.;
		}
	}
	double Pmean(P_tot/(double)(nx*ny*nz));
	std::cerr << " (mean # of points per mesh: " << Pmean << ")";
	double inv_Pmean(1./Pmean);
	this->multiply_data_all((float)inv_Pmean);
	this->subtract_data_all(1.);
	std::cerr << " done." << std::endl;
	ngal = P_tot;
}

void FieldData::assignment_nonorm(std::vector<myhosthalo_str> &D,double nh,bool CIC,bool interlace){

	int ntake((int)(nh*Lx*Ly*Lz));

	if(CIC){
		std::cerr << "start CIC density assignment of " << ntake << " points to ("
		<< nx << "," << ny << "," << nz << ") grid points" << std::endl;
	}else{
		std::cerr << "start NGP density assignment of " << ntake << " points to ("
		<< nx << "," << ny << "," << nz << ") grid points" << std::endl;
	}
	double P_tot(0);

	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		int i1, i2, i3;
		int i1p, i2p, i3p;
		float u, v, w;
		#pragma omp for schedule(guided)
		for(int n=0;n<ntake;n++){

			if(CIC){
				u = (nx*D[n].pos[0])/Lx;
				v = (ny*D[n].pos[1])/Ly;
				w = (nz*D[n].pos[2])/Lz;
			}else{ // FOR NGP
				u = (nx*D[n].pos[0])/Lx+0.5;
				v = (ny*D[n].pos[1])/Ly+0.5;
				w = (nz*D[n].pos[2])/Lz+0.5;
			}
			if(interlace){
				u += 0.5;
				v += 0.5;
				w += 0.5;
			}
			i1 = floor(u);
			i2 = floor(v);
			i3 = floor(w);

			if(interlace){
				i1 -= 1;
				i2 -= 1;
				i3 -= 1;
			}
			while(i1 >= (int)nx) i1 -= (int)nx;
			while(i2 >= (int)ny) i2 -= (int)ny;
			while(i3 >= (int)nz) i3 -= (int)nz;
			while(i1 < 0) i1 += (int)nx;
			while(i2 < 0) i2 += (int)ny;
			while(i3 < 0) i3 += (int)nz;

			if(!CIC){
				this->add_data( i1,  i2,  i3, 1.);
			}else{
				u -= i1;
				v -= i2;
				w -= i3;

				i1p = i1 + 1;
				i2p = i2 + 1;
				i3p = i3 + 1;

				if(i1p >= (int)nx) i1p -= (int)nx;
				if(i2p >= (int)ny) i2p -= (int)ny;
				if(i3p >= (int)nz) i3p -= (int)nz;

				this->add_data( i1,  i2,  i3, (1.-u)*(1.-v)*(1.-w));
				this->add_data( i1,  i2, i3p, (1.-u)*(1.-v)*(w)   );
				this->add_data( i1, i2p,  i3, (1.-u)*(v)   *(1.-w));
				this->add_data( i1, i2p, i3p, (1.-u)*(v)   *(w)   );
				this->add_data(i1p,  i2,  i3, (u)   *(1.-v)*(1.-w));
				this->add_data(i1p,  i2, i3p, (u)   *(1.-v)*(w)   );
				this->add_data(i1p, i2p,  i3, (u)   *(v)   *(1.-w));
				this->add_data(i1p, i2p, i3p, (u)   *(v)   *(w)   );
			}
			#pragma omp atomic
			P_tot += 1.;
		}
	}
	// double Pmean(P_tot/(double)(nx*ny*nz));
	// std::cout << " (mean # of points per mesh: " << Pmean << ")";
	// double inv_Pmean(1./Pmean);
	// this->multiply_data_all((float)inv_Pmean);
	// this->subtract_data_all(1.);
	std::cerr << " done." << std::endl;
	ngal = P_tot;
}

void FieldData::assignment_nonorm(std::vector<myhosthalo_str> &D,double nh_min,double nh_max,bool CIC,bool interlace){

	int ntake_min((int)(nh_min*Lx*Ly*Lz));
	int ntake_max((int)(nh_max*Lx*Ly*Lz));

	if(CIC){
		std::cerr << "start CIC density assignment of " << ntake_max-ntake_min << " points to a ("
		<< nx << "," << ny << "," << nz << ") grid points" << std::endl;
	}else{
		std::cerr << "start NGP density assignment of " << ntake_max-ntake_min << " points to a ("
		<< nx << "," << ny << "," << nz << ") grid points" << std::endl;
	}
	double P_tot(0);

	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		int i1, i2, i3;
		int i1p, i2p, i3p;
		float u, v, w;
		#pragma omp for schedule(guided)
		for(int n=ntake_min;n<ntake_max;n++){

			if(CIC){
				u = (nx*D[n].pos[0])/Lx;
				v = (ny*D[n].pos[1])/Ly;
				w = (nz*D[n].pos[2])/Lz;
			}else{ // FOR NGP
				u = (nx*D[n].pos[0])/Lx+0.5;
				v = (ny*D[n].pos[1])/Ly+0.5;
				w = (nz*D[n].pos[2])/Lz+0.5;
			}
			if(interlace){
				u += 0.5;
				v += 0.5;
				w += 0.5;
			}
			i1 = floor(u);
			i2 = floor(v);
			i3 = floor(w);

			if(interlace){
				i1 -= 1;
				i2 -= 1;
				i3 -= 1;
			}
			while(i1 >= (int)nx) i1 -= (int)nx;
			while(i2 >= (int)ny) i2 -= (int)ny;
			while(i3 >= (int)nz) i3 -= (int)nz;
			while(i1 < 0) i1 += (int)nx;
			while(i2 < 0) i2 += (int)ny;
			while(i3 < 0) i3 += (int)nz;

			if(!CIC){
				this->add_data( i1,  i2,  i3, 1.);
			}else{
				u -= i1;
				v -= i2;
				w -= i3;

				i1p = i1 + 1;
				i2p = i2 + 1;
				i3p = i3 + 1;

				if(i1p >= (int)nx) i1p -= (int)nx;
				if(i2p >= (int)ny) i2p -= (int)ny;
				if(i3p >= (int)nz) i3p -= (int)nz;

				this->add_data( i1,  i2,  i3, (1.-u)*(1.-v)*(1.-w));
				this->add_data( i1,  i2, i3p, (1.-u)*(1.-v)*(w)   );
				this->add_data( i1, i2p,  i3, (1.-u)*(v)   *(1.-w));
				this->add_data( i1, i2p, i3p, (1.-u)*(v)   *(w)   );
				this->add_data(i1p,  i2,  i3, (u)   *(1.-v)*(1.-w));
				this->add_data(i1p,  i2, i3p, (u)   *(1.-v)*(w)   );
				this->add_data(i1p, i2p,  i3, (u)   *(v)   *(1.-w));
				this->add_data(i1p, i2p, i3p, (u)   *(v)   *(w)   );
			}
			#pragma omp atomic
			P_tot += 1.;
		}
	}
	// double Pmean(P_tot/(double)(nx*ny*nz));
	// std::cout << " (mean # of points per mesh: " << Pmean << ")";
	// double inv_Pmean(1./Pmean);
	// this->multiply_data_all((float)inv_Pmean);
	// this->subtract_data_all(1.);
	std::cerr << " done." << std::endl;
	ngal = P_tot;
}

void FieldData::assignment(std::vector<myhosthalo_str> &D,bool CIC,bool interlace){

	if(CIC){
		std::cerr << "start CIC density assignment of " << D.size() << " points to a ("
		<< nx << "," << ny << "," << nz << ") grid points" << std::endl;
	}else{
		std::cerr << "start NGP density assignment of " << D.size() << " points to a ("
		<< nx << "," << ny << "," << nz << ") grid points" << std::endl;
	}
	double P_tot(0);

	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		int i1, i2, i3;
		int i1p, i2p, i3p;
		float u, v, w;
		#pragma omp for schedule(guided)
		for(int n=0;n<D.size();n++){
			if(CIC){
				u = (nx*D[n].pos[0])/Lx;
				v = (ny*D[n].pos[1])/Ly;
				w = (nz*D[n].pos[2])/Lz;
			}else{ // FOR NGP
				u = (nx*D[n].pos[0])/Lx+0.5;
				v = (ny*D[n].pos[1])/Ly+0.5;
				w = (nz*D[n].pos[2])/Lz+0.5;
			}
			if(interlace){
				u += 0.5;
				v += 0.5;
				w += 0.5;
			}
			i1 = floor(u);
			i2 = floor(v);
			i3 = floor(w);

			if(interlace){
				i1 -= 1;
				i2 -= 1;
				i3 -= 1;
			}
			while(i1 >= (int)nx) i1 -= (int)nx;
			while(i2 >= (int)ny) i2 -= (int)ny;
			while(i3 >= (int)nz) i3 -= (int)nz;
			while(i1 < 0) i1 += (int)nx;
			while(i2 < 0) i2 += (int)ny;
			while(i3 < 0) i3 += (int)nz;

			if(!CIC){
				this->add_data( i1,  i2,  i3, 1.);
			}else{
				u -= i1;
				v -= i2;
				w -= i3;

				i1p = i1 + 1;
				i2p = i2 + 1;
				i3p = i3 + 1;

				if(i1p >= (int)nx) i1p -= (int)nx;
				if(i2p >= (int)ny) i2p -= (int)ny;
				if(i3p >= (int)nz) i3p -= (int)nz;

				this->add_data( i1,  i2,  i3, (1.-u)*(1.-v)*(1.-w));
				this->add_data( i1,  i2, i3p, (1.-u)*(1.-v)*(w)   );
				this->add_data( i1, i2p,  i3, (1.-u)*(v)   *(1.-w));
				this->add_data( i1, i2p, i3p, (1.-u)*(v)   *(w)   );
				this->add_data(i1p,  i2,  i3, (u)   *(1.-v)*(1.-w));
				this->add_data(i1p,  i2, i3p, (u)   *(1.-v)*(w)   );
				this->add_data(i1p, i2p,  i3, (u)   *(v)   *(1.-w));
				this->add_data(i1p, i2p, i3p, (u)   *(v)   *(w)   );
			}
			#pragma omp atomic
			P_tot += 1.;
		}
	}

	double Pmean(P_tot/(double)(nx*ny*nz));
	std::cerr << " (mean # of points per mesh: " << Pmean << ")";
	double inv_Pmean(1./Pmean);
	this->multiply_data_all((float)inv_Pmean);
	this->subtract_data_all(1.);
	std::cerr << " done." << std::endl;
	ngal = P_tot;
}

void FieldData::assignment(std::vector<myhosthalo_str> &D,bool CIC,bool interlace, double sfac, int los_dir){ // dir: line-of-sight direction (0: x, 1: y, 2: z)

	if(CIC){
		std::cerr << "start CIC density assignment of " << D.size() << " points to a ("
		<< nx << "," << ny << "," << nz << ") grid points in redshift space" << std::endl;
	}else{
		std::cerr << "start NGP density assignment of " << D.size() << " points to a ("
		<< nx << "," << ny << "," << nz << ") grid points in redshift space" << std::endl;
	}
	double P_tot(0);

	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		int i1, i2, i3;
		int i1p, i2p, i3p;
		float u, v, w;
		#pragma omp for schedule(guided)
		for(int n=0;n<D.size();n++){
			if(CIC){
				if(los_dir==0){
	                u = (nx*(D[n].pos[0]+sfac*D[n].vel[0]))/Lx;
					v = (ny*D[n].pos[1])/Ly;
					w = (nz*D[n].pos[2])/Lz;
				}else if(los_dir==1){
					u = (nx*D[n].pos[0])/Lx;
					v = (ny*(D[n].pos[1]+sfac*D[n].vel[1]))/Ly;
					w = (nz*D[n].pos[2])/Lz;
				}else if(los_dir==2){
					u = (nx*D[n].pos[0])/Lx;
					v = (ny*D[n].pos[1])/Ly;
					w = (nz*(D[n].pos[2]+sfac*D[n].vel[2]))/Lz;
				}else{
					std::cerr << "Invalid los direction!!" << std::endl;
				}
			}else{ // FOR NGP
				if(los_dir==0){
					u = (nx*(D[n].pos[0]+sfac*D[n].vel[0]))/Lx+0.5;
					v = (ny*D[n].pos[1])/Ly+0.5;
					w = (nz*D[n].pos[2])/Lz+0.5;
				}else if(los_dir==1){
					u = (nx*D[n].pos[0])/Lx+0.5;
					v = (ny*(D[n].pos[1]+sfac*D[n].vel[1]))/Ly+0.5;
					w = (nz*D[n].pos[2])/Lz+0.5;
				}else if(los_dir==2){
					u = (nx*D[n].pos[0])/Lx+0.5;
					v = (ny*D[n].pos[1])/Lz+0.5;
					w = (nz*(D[n].pos[2]+sfac*D[n].vel[2]))/Lz+0.5;
				}else{
					std::cerr << "Invalid los direction!!" << std::endl;
				}
			}
			if(interlace){
				u += 0.5;
				v += 0.5;
				w += 0.5;
			}
			if(!CIC){
				i1 = floor(u);
				i2 = floor(v);
				i3 = floor(w);
				while(i1 >= (int)nx) i1 -= (int)nx;
				while(i2 >= (int)ny) i2 -= (int)ny;
				while(i3 >= (int)nz) i3 -= (int)nz;
				while(i1 < 0) i1 += (int)nx;
				while(i2 < 0) i2 += (int)ny;
				while(i3 < 0) i3 += (int)nz;
				this->add_data( i1,  i2,  i3, 1.);
			}else{
				i1 = floor(u); u -= i1;
                        	i2 = floor(v); v -= i2;
                        	i3 = floor(w); w -= i3;

				if(interlace){
					i1 -= 1;
					i2 -= 1;
					i3 -= 1;
				}
				while(i1 >= (int)nx) i1 -= (int)nx;
				while(i2 >= (int)ny) i2 -= (int)ny;
				while(i3 >= (int)nz) i3 -= (int)nz;
				while(i1 < 0) i1 += (int)nx;
				while(i2 < 0) i2 += (int)ny;
				while(i3 < 0) i3 += (int)nz;

				i1p = i1 + 1;
				i2p = i2 + 1;
				i3p = i3 + 1;

				if(i1p >= (int)nx) i1p -= (int)nx;
				if(i2p >= (int)ny) i2p -= (int)ny;
				if(i3p >= (int)nz) i3p -= (int)nz;

				this->add_data( i1,  i2,  i3, (1.-u)*(1.-v)*(1.-w));
				this->add_data( i1,  i2, i3p, (1.-u)*(1.-v)*(w)   );
				this->add_data( i1, i2p,  i3, (1.-u)*(v)   *(1.-w));
				this->add_data( i1, i2p, i3p, (1.-u)*(v)   *(w)   );
				this->add_data(i1p,  i2,  i3, (u)   *(1.-v)*(1.-w));
				this->add_data(i1p,  i2, i3p, (u)   *(1.-v)*(w)   );
				this->add_data(i1p, i2p,  i3, (u)   *(v)   *(1.-w));
				this->add_data(i1p, i2p, i3p, (u)   *(v)   *(w)   );
			}
			#pragma omp atomic
			P_tot += 1.;
		}
	}

	double Pmean(P_tot/(double)(nx*ny*nz));
	std::cerr << " (mean # of points per mesh: " << Pmean << ")";
	double inv_Pmean(1./Pmean);
	this->multiply_data_all((float)inv_Pmean);
	this->subtract_data_all(1.);
	std::cerr << " done." << std::endl;
	ngal = P_tot;
}



void FieldData::average2fields (FieldData &f1, FieldData &f2){
	ngal = f1.get_ngal();
	#pragma omp parallel for
	for(long long int a=0;a<2*(long long int)nx*(long long int)ny*((long long int)nz/2+1);a++)
	this->data[a] = (f1.data[a]+f2.data[a])/2.;
}

void FieldData::adjust_grid(void){
	float delta_grid_x = M_PI / (float)nx;
	float delta_grid_y = M_PI / (float)ny;
	float delta_grid_z = M_PI / (float)nz;
	#pragma omp parallel for
	for(int a=0;a<nx;a++){
		float aa = (a<nx/2)?(float)a:(float)(a-(int)nx);
		for(int b=0;b<ny;b++){
			float bb = (b<ny/2)?(float)b:(float)(b-(int)ny);
			for(int c=0;c<nz/2+1;c++){
				float cc = (c<nz/2)?(float)c:(float)(c-(int)nz);
				//float ksum = aa + bb + cc;
				//float fac_re = cos(ksum*delta_grid);
				//float fac_im = -sin(ksum*delta_grid);
				float phase_shift(aa*delta_grid_x+bb*delta_grid_y+cc*delta_grid_z);
				float fac_re(cos(phase_shift));
				float fac_im(-sin(phase_shift));
				long long int bin(((long long int)(a*ny+b))*(2*(nz/2+1))+2*c);
				float tmp_re(this->data[bin] * fac_re - this->data[bin+1] * fac_im);
				float tmp_im(this->data[bin+1] * fac_re + this->data[bin] * fac_im);
				this->data[bin] = tmp_re;
				this->data[bin+1] = tmp_im;
			}
		}
	}
}

void FieldData::apply_filter(float R, std::string filter_type){
	float kfundx = 2.*M_PI/Lx;
	float kfundy = 2.*M_PI/Ly;
	float kfundz = 2.*M_PI/Lz;
	#pragma omp parallel
	{
		float ii, jj, kk;
		float knorm;
		float w;

		#pragma omp for schedule(guided)
		for(long long int i=0;i<(int)nx;i++){
			float ii = (i>(int)nx/2)?i-(int)nx:i;
			for(long long int j=0;j<(int)ny;j++){
				float jj = (j>(int)ny/2)?j-(int)ny:j;
				for(long long int k=0;k<=(int)nz/2;k++){
					float kk = (k>(int)nz/2)?k-(int)nz:k;
					float knorm = sqrt(kfundx*kfundx*ii*ii+kfundy*kfundy*jj*jj+kfundz*kfundz*kk*kk);
					if(filter_type=="gauss") w = Gwindow(R*knorm);
					if(filter_type=="tophat") w = THwindow(R*knorm);
					if(filter_type=="sharpk") w = Sharpk(R*knorm);
					long long int bin = (i*(long long int)ny+j)*(2*((long long int)nz/2+1))+2*k;
					this->data[bin] *= w;
					this->data[bin+1] *= w;
				}
			}
		}
	}

}

void FieldData::multiply_sqrtPk(std::string tfname, double redshift, double As, double ns, double k0){
	bool space_org(this->inFourierSpace);
	if(!space_org) this->do_fft();
	double kfundx(2.*M_PI/Lx);
	double kfundy(2.*M_PI/Ly);
	double kfundz(2.*M_PI/Lz);
	std::cerr << "Multiply transfer function ..." << std::endl;
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		TransferFunction tr(tfname,"cb");
		#pragma omp for schedule(auto)
		for(long long int i=0;i<nx;i++){
			if(thistask == 0) std::cerr << i << "\r" << std::flush;
			double ii((i<(int)nx/2)?i:i-(int)nx);
			for(long long int j=0;j<(int)ny;j++){
				double jj((j<(int)ny/2)?j:j-(int)ny);
				for(long long int k=0;k<(int)nz/2+1;k++){
					double kk((k<(int)nz/2)?k:k-(int)nz);
					double kwave(sqrt(kfundx*kfundx*ii*ii + kfundy*kfundy*jj*jj + kfundz*kfundz*kk*kk));
					double p_of_k = 2.*M_PI*M_PI / (kwave * kwave * kwave) * As * pow(kwave/(k0),ns-1.);
					double trans = tr.get_transfer(kwave,redshift);
					data[(i*(long long int)ny+j)*(2*((long long int)nz/2+1))+(2*k)  ] *= sqrt(p_of_k)*trans;
					data[(i*(long long int)ny+j)*(2*((long long int)nz/2+1))+(2*k)+1] *= sqrt(p_of_k)*trans;
				}
			}
		}
	}
	data[0] = 0.;
	data[1] = 0.;
	std::cerr << "Done." << std::endl;

	if(!space_org) this->do_ifft();
}

void FieldData::multiply_transfer(TransferFunction& tr, double redshift){
	bool space_org(this->inFourierSpace);
	if(!space_org) this->do_fft();
	double kfundx(2.*M_PI/Lx);
	double kfundy(2.*M_PI/Ly);
	double kfundz(2.*M_PI/Lz);
	double trans;
	std::cerr << "Multiply transfer function ..." << std::endl;
	#pragma omp parallel for schedule(auto)
	for(int i=0;i<nx;i++){
		std::cerr << i << "\r" << std::flush;
		double ii((i<(int)nx/2)?i:i-(int)nx);
		for(int j=0;j<(int)ny;j++){
			double jj((j<(int)ny/2)?j:j-(int)ny);
			for(int k=0;k<(int)nz/2+1;k++){
				double kk((k<(int)nz/2)?k:k-(int)nz);
				double kwave(sqrt(kfundx*kfundx*ii*ii + kfundy*kfundy*jj*jj + kfundz*kfundz*kk*kk));
				trans = tr.get_transfer(kwave,redshift);
				data[(i*ny+j)*(2*(nz/2+1))+(2*k)  ] *= trans;
				data[(i*ny+j)*(2*(nz/2+1))+(2*k)+1] *= trans;
			}
		}
	}
	data[0] = 0.;
	data[1] = 0.;
	std::cerr << "Done." << std::endl;
	if(!space_org) this->do_ifft();
}

void FieldData::divide_transfer(TransferFunction& tr, double redshift){
	bool space_org(this->inFourierSpace);
	if(!space_org) this->do_fft();
	double kfundx(2.*M_PI/Lx);
	double kfundy(2.*M_PI/Ly);
	double kfundz(2.*M_PI/Lz);
	double trans;
	std::cerr << "Multiply transfer function ..." << std::endl;
	#pragma omp parallel for schedule(auto)
	for(int i=0;i<nx;i++){
		std::cerr << i << "\r" << std::flush;
		double ii((i<(int)nx/2)?i:i-(int)nx);
		for(int j=0;j<(int)ny;j++){
			double jj((j<(int)ny/2)?j:j-(int)ny);
			for(int k=0;k<(int)nz/2+1;k++){
				double kk((k<(int)nz/2)?k:k-(int)nz);
				double kwave(sqrt(kfundx*kfundx*ii*ii + kfundy*kfundy*jj*jj + kfundz*kfundz*kk*kk));
				trans = tr.get_transfer(kwave,redshift);
				data[(i*ny+j)*(2*(nz/2+1))+(2*k)  ] /= trans;
				data[(i*ny+j)*(2*(nz/2+1))+(2*k)+1] /= trans;
			}
		}
	}
	data[0] = 0.;
	data[1] = 0.;
	std::cerr << "Done." << std::endl;
	if(!space_org) this->do_ifft();
}




void FieldData::load(std::string ifname){
	std::ifstream fin(ifname.c_str());
	fin.read((char *)&data[0],sizeof(float)*2*(long long int)nx*(long long int)ny*((long long int)nz/2+1));
	fin.close();

}

void FieldData::save_slice(std::string ofname,int zpixs){
	std::ofstream fout(ofname.c_str());
	for(long long int i=0;i<nx;i++){
		for(long long int j=0;j<ny;j++){
			float overdens(0);
			for(long long int k=0;k<zpixs;k++){
				overdens += data[(i*(long long int)ny+j)*(2*((long long int)nz/2+1))+k]/(float)zpixs;
			}
			fout.write((char *)&overdens,sizeof(float));
		}
	}
	fout.close();

}


float THwindow(float kR){
	if(kR == 0) {
		return 1.;
	} else if(kR>0. && kR <= 0.2) { return  1.- kR*kR/10. + kR*kR*kR*kR/280. ;
	} else {
		return 3.*(sin(kR)-kR*cos(kR))/(kR*kR*kR);
	}
}

float Sharpk(float kR){
	if(kR <= 1. ) return 1.;
	else return 0.;
}

float Gwindow(float kR){
	if(kR == 0) return 1.;
	else return exp(-0.5*kR*kR);
}

BinnedData FieldData::calc_power_nowindow(int nbin, float kmin,float kmax,bool logbin,int ell) const {
	BinnedData powerspec(nbin,kmin,kmax,logbin);

	float kfund_x = 2.*M_PI/(Lx);
	float kfund_y = 2.*M_PI/(Ly);
	float kfund_z = 2.*M_PI/(Lz);
	float knyq_x = kfund_x * nx/2.;
	float knyq_y = kfund_y * ny/2.;
	float knyq_z = kfund_z * nz/2.;
	float knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	float vol(Lx*Ly*Lz);
	//std::cerr << "Fundamental wave numbers: " << kfund_x << "  " << kfund_y << "  " << kfund_z << std::endl;
	//std::cerr << "Number of grids: " << nx << "  " << ny << "  " << nz << std::endl;
	//std::cerr << "Volume: " << vol << std::endl;
	//std::cerr << "Bins: " << nbin << "bins in [" << kmin << ", " << kmax << "), logartithm?" << logbin << std::endl;
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float kx, ky, kz;
		float fre, fim;
		float knorm, pnorm, mu;
		float weight;
		BinnedData local_pow = powerspec;
		#pragma omp for
		for(int i=0;i<nx;i++){
			ii = (i>nx/2)?i-(int)nx:i;
			kx = kfund_x * ii;
			if(fabs(kx)>powerspec.get_max()) continue;
			if(fabs(kx)>knyq_x) continue;
			for(int j=0;j<ny;j++){
				jj = (j>ny/2)?j-(int)ny:j;
				ky = kfund_y * jj;
				if(fabs(ky)>powerspec.get_max()) continue;
				if(fabs(ky)>knyq_y) continue;
				for(int k=0;k<nz/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>nz/2)?k-(int)nz:k;
					kz = kfund_z * kk;
					if(fabs(kz)>powerspec.get_max()) continue;
					if(fabs(kz)>knyq_z) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>powerspec.get_max()) continue;
					if(knorm>knyq_min) continue;
					mu = kz/knorm;
					fre = this->get_data_re(i,j,k);
					fim = this->get_data_im(i,j,k);
					pnorm = vol*(fre*fre+fim*fim);
					//pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim)-shotnoise);
					pnorm *= (2.*ell+1.)*gsl_sf_legendre_Pl(ell, mu);
					if(k==0) weight = 0.5;
					else weight = 1.;
					local_pow.push_data(knorm,pnorm,weight);
					//powerspec.push_data(knorm,pnorm,weight);
				}
			}
		}
		#pragma omp critical
		{
			powerspec = powerspec+local_pow;
		}
	}
	return powerspec;
}

BinnedData FieldData::calc_power_noshot(int nbin, float kmin,float kmax,bool logbin,int ell) const {
	BinnedData powerspec(nbin,kmin,kmax,logbin);

	float kfund_x = 2.*M_PI/(Lx);
	float kfund_y = 2.*M_PI/(Ly);
	float kfund_z = 2.*M_PI/(Lz);
	float knyq_x = kfund_x * nx/2.;
	float knyq_y = kfund_y * ny/2.;
	float knyq_z = kfund_z * nz/2.;
	float knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	float vol(Lx*Ly*Lz);
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float kx, ky, kz;
		float fre, fim;
		float knorm, pnorm, mu;
		float weight;
		float wx, wy, wz;
		BinnedData local_pow = powerspec;
		#pragma omp for
		for(int i=0;i<nx;i++){
			ii = (i>nx/2)?i-(int)nx:i;
			kx = kfund_x * ii;
			if(fabs(kx)>powerspec.get_max()) continue;
			if(fabs(kx)>knyq_x) continue;
			wx = 1.0/gsl_sf_sinc(ii/((float)nx));
			wx *= wx; // NGP
			wx *= wx; // CIC
			for(int j=0;j<ny;j++){
				jj = (j>ny/2)?j-(int)ny:j;
				ky = kfund_y * jj;
				if(fabs(ky)>powerspec.get_max()) continue;
				if(fabs(ky)>knyq_y) continue;
				wy = 1.0/gsl_sf_sinc(jj/((float)ny));
				wy *= wy; // NGP
				wy *= wy; // CIC
				for(int k=0;k<nz/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>nz/2)?k-(int)nz:k;
					kz = kfund_z * kk;
					if(fabs(kz)>powerspec.get_max()) continue;
					if(fabs(kz)>knyq_z) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>powerspec.get_max()) continue;
					if(knorm>knyq_min) continue;
					wz = 1.0/gsl_sf_sinc(kk/((float)nz));
					wz *= wz; // NGP
					wz *= wz; // CIC
					mu = kz/knorm;
					fre = this->get_data_re(i,j,k);
					fim = this->get_data_im(i,j,k);
					pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim));
					pnorm *= (2.*ell+1.)*gsl_sf_legendre_Pl(ell, mu);
					if(k==0) weight = 0.5;
					else weight = 1.;
					local_pow.push_data(knorm,pnorm,weight);
					// local_pow.push_data_CIC(knorm,pnorm,weight);

					//powerspec.push_data(knorm,pnorm,weight);
				}
			}
		}
		#pragma omp critical
		{
			powerspec = powerspec+local_pow;
		}
	}
	return powerspec;
}

BinnedData FieldData::calc_power(int nbin, float kmin, float kmax, bool logbin, int ell) const {
	BinnedData powerspec(nbin,kmin,kmax,logbin);
	double redshift = 0.38;
	double Omegam = 0.23108619451522827;
	double Omegam_fid = 0.31;
	double DAtarget = ZtoComovingD_LCDM(redshift,Omegam);
	double DAfid = ZtoComovingD_LCDM(redshift,Omegam_fid);
	double Htarget = Hz_LCDM(redshift,Omegam);
	double Hfid = Hz_LCDM(redshift,Omegam_fid);
	float kfund_x = 2.*M_PI/(Lx * DAfid/DAtarget);
	float kfund_y = 2.*M_PI/(Ly * DAfid/DAtarget);
	float kfund_z = 2.*M_PI/(Lz * Htarget/Hfid);
	float knyq_x = kfund_x * nx/2.;
	float knyq_y = kfund_y * ny/2.;
	float knyq_z = kfund_z * nz/2.;
	float knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	float vol((Lx*DAfid/DAtarget)*(Ly*DAfid/DAtarget)*(Lz*Htarget/Hfid));
	float shotnoise = vol/(float)ngal;
	//std::cerr << "Fundamental wave numbers: " << kfund_x << "  " << kfund_y << "  " << kfund_z << std::endl;
	//std::cerr << "Number of grids: " << nx << "  " << ny << "  " << nz << std::endl;
	//std::cerr << "Volume: " << vol << std::endl;
	//std::cerr << "Bins: " << nbin << "bins in [" << kmin << ", " << kmax << "), logartithm?" << logbin << std::endl;
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float kx, ky, kz;
		float fre, fim;
		float knorm, pnorm, mu;
		float weight;
		float wx, wy, wz;
		float wx2, wy2, wz2;
		BinnedData local_pow = powerspec;
		#pragma omp for
		for(int i=0;i<nx;i++){
			ii = (i>nx/2)?i-(int)nx:i;
			kx = kfund_x * ii;
			if(fabs(kx)>powerspec.get_max()) continue;
			if(fabs(kx)>knyq_x) continue;
			wx = 1.0/gsl_sf_sinc(ii/((float)nx));
			wx *= wx; // NGP
			wx *= wx; // CIC
			wx2 = sin(M_PI * ii/((float)nx));
			wx2 = 1. - 2/3. * (wx2*wx2);       // shotnoise factor for CIC assignment
			//wx2 = cos(M_PI * ii/((float)nx));
			//wx2 = (1.+2*wx2+wx2*wx2)*(2+wx2)/12.; // shotnoise factor for interlaced CIC
			for(int j=0;j<ny;j++){
				jj = (j>ny/2)?j-(int)ny:j;
				ky = kfund_y * jj;
				if(fabs(ky)>powerspec.get_max()) continue;
				if(fabs(ky)>knyq_y) continue;
				wy = 1.0/gsl_sf_sinc(jj/((float)ny));
				wy *= wy; // NGP
				wy *= wy; // CIC
				wy2 = sin(M_PI * jj/((float)ny));
				wy2 = 1. - 2/3. * (wy2*wy2);       // shotnoise factor for CIC assignment
				//wy2 = cos(M_PI * jj/((float)ny));
				//wy2 = (1.+2*wy2+wy2*wy2)*(2+wy2)/12.; // shotnoise factor for interlaced CIC
				for(int k=0;k<nz/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>nz/2)?k-(int)nz:k;
					kz = kfund_z * kk;
					if(fabs(kz)>powerspec.get_max()) continue;
					if(fabs(kz)>knyq_z) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>powerspec.get_max()) continue;
					if(knorm>knyq_min) continue;
					wz = 1.0/gsl_sf_sinc(kk/((float)nz));
					wz *= wz; // NGP
					wz *= wz; // CIC
					wz2 = sin(M_PI * kk/((float)nz));
					wz2 = 1. - 2/3. * (wz2*wz2);       // shotnoise factor for CIC assignment
					//wz2 = cos(M_PI * kk/((float)nz));
					//wz2 = (1.+2*wz2+wz2*wz2)*(2+wz2)/12.; // shotnoise factor for interlaced CIC
					mu = kz/knorm;
					fre = this->get_data_re(i,j,k);
					fim = this->get_data_im(i,j,k);
					pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim)-shotnoise*wx2*wy2*wz2);
					//pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim)-shotnoise);
					pnorm *= (2.*ell+1.)*gsl_sf_legendre_Pl(ell, mu);
					if(k==0) weight = 0.5;
					else weight = 1.;
					local_pow.push_data(knorm,pnorm,weight);
					//powerspec.push_data(knorm,pnorm,weight);
				}
			}
		}
		#pragma omp critical
		{
			powerspec = powerspec+local_pow;
		}
	}
	return powerspec;
}

// BinnedData FieldData::calc_power(int nbin, float kmin,float kmax,bool logbin,int ell,float Omegam,float Omegam_fid,float redshift) const {
BinnedData FieldData::calc_power(int nbin, float kmin,float kmax,bool logbin,int ell,std::string fname,float Omegam_fid,float redshift, int los_dir) const {
	BinnedData powerspec(nbin,kmin,kmax,logbin);

	//float DAtrue(ZtoComovingD(redshift,Omegam));
        float DAtarget(ZtoComovingD(redshift,fname));
	float DAfid(ZtoComovingD_LCDM(redshift,Omegam_fid));
	//float Htrue(Hz(redshift,Omegam));
	float Htarget(Hz(redshift,fname));
	float Hfid(Hz_LCDM(redshift,Omegam_fid));
	if(redshift < 1e-4) DAtarget = DAfid = 1.;

	float kfund_x = 2.*M_PI/(Lx*DAfid/DAtarget);
	float kfund_y = 2.*M_PI/(Ly*DAfid/DAtarget);
	float kfund_z = 2.*M_PI/(Lz*DAfid/DAtarget);

	switch (los_dir){
		case 0:
			kfund_x = 2.*M_PI/(Lx*Htarget/Hfid);
			break;
		case 1:
			kfund_y = 2.*M_PI/(Ly*Htarget/Hfid);
			break;
		case 2:
			kfund_z = 2.*M_PI/(Lz*Htarget/Hfid);
			break;
		default:
			kfund_z = 2.*M_PI/(Lz*Htarget/Hfid);
			break;
	}
	float knyq_x = kfund_x * nx/2.;
	float knyq_y = kfund_y * ny/2.;
	float knyq_z = kfund_z * nz/2.;
	float knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	float vol(Lx*DAfid/DAtarget*Ly*DAfid/DAtarget*Lz*Htarget/Hfid);
	float shotnoise = vol/(float)ngal;
	//std::cerr << "Fundamental wave numbers: " << kfund_x << "  " << kfund_y << "  " << kfund_z << std::endl;
	//std::cerr << "Number of grids: " << nx << "  " << ny << "  " << nz << std::endl;
	//std::cerr << "Volume: " << vol << std::endl;
	//std::cerr << "Bins: " << nbin << "bins in [" << kmin << ", " << kmax << "), logartithm?" << logbin << std::endl;
	//#pragma omp parallel
	{
		//	int ntasks = omp_get_num_threads();
		//	int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float kx, ky, kz;
		float fre, fim;
		float knorm, pnorm, mu;
		float weight;
		float wx, wy, wz;
		float wx2, wy2, wz2;
		//BinnedData local_pow = powerspec;
		//#pragma omp for
		for(int i=0;i<nx;i++){
			ii = (i>nx/2)?i-(int)nx:i;
			kx = kfund_x * ii;
			if(fabs(kx)>powerspec.get_max()) continue;
			if(fabs(kx)>knyq_x) continue;
			wx = 1.0/gsl_sf_sinc(ii/((float)nx));
			wx *= wx; // NGP
			wx *= wx; // CIC
			wx2 = cos(M_PI * ii/((float)nx));
			wx2 = (1.+2*wx2+wx2*wx2)*(2+wx2)/12.;
			for(int j=0;j<ny;j++){
				jj = (j>ny/2)?j-(int)ny:j;
				ky = kfund_y * jj;
				if(fabs(ky)>powerspec.get_max()) continue;
				if(fabs(ky)>knyq_y) continue;
				wy = 1.0/gsl_sf_sinc(jj/((float)ny));
				wy *= wy; // NGP
				wy *= wy; // CIC
				wy2 = cos(M_PI * jj/((float)ny));
				wy2 = (1.+2*wy2+wy2*wy2)*(2+wy2)/12.;
				for(int k=0;k<(int)nz/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>nz/2)?k-(int)nz:k;
					kz = kfund_z * kk;
					if(fabs(kz)>powerspec.get_max()) continue;
					if(fabs(kz)>knyq_z) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>powerspec.get_max()) continue;
					if(knorm>knyq_min) continue;
					wz = 1.0/gsl_sf_sinc(kk/((float)nz));
					wz *= wz; // NGP
					wz *= wz; // CIC
					wz2 = cos(M_PI * kk/((float)nz));
					wz2 = (1.+2*wz2+wz2*wz2)*(2+wz2)/12.;
					switch (los_dir){
						case 0:
							mu = kx/knorm;
							break;
						case 1:
							mu = ky/knorm;
							break;
						case 2:
							mu = kz/knorm;
							break;
						default:
							mu = kz/knorm;
							break;
					}
					fre = this->get_data_re(i,j,k);
					fim = this->get_data_im(i,j,k);
					pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim)-shotnoise*wx2*wy2*wz2);
					pnorm *= (2.*ell+1.)*gsl_sf_legendre_Pl(ell, mu);
					if(k==0) weight = 0.5;
					else weight = 1.;
					//local_pow.push_data(knorm,pnorm,weight);
					powerspec.push_data(knorm,pnorm,weight);
				}
			}
		}
		//#pragma omp critical
		{
			//powerspec = powerspec+local_pow;
		}
	}
	return powerspec;
}

void FieldData::calc_power(BinnedData &powerspec,int ell,std::string fname,float Omegam_fid,float redshift, int los_dir) const {
	// BinnedData powerspec(nbin,kmin,kmax,logbin);

	int nbin = powerspec.get_nbin();
	double kmin = powerspec.get_min();
	double kmax = powerspec.get_max();
	bool logbin = powerspec.get_logbin();

	//float DAtrue(ZtoComovingD(redshift,Omegam));
        float DAtarget(ZtoComovingD(redshift,fname));
	float DAfid(ZtoComovingD_LCDM(redshift,Omegam_fid));
	//float Htrue(Hz(redshift,Omegam));
	float Htarget(Hz(redshift,fname));
	float Hfid(Hz_LCDM(redshift,Omegam_fid));
	if(redshift < 1e-4) DAtarget = DAfid = 1.;

	float kfund_x = 2.*M_PI/(Lx*DAfid/DAtarget);
	float kfund_y = 2.*M_PI/(Ly*DAfid/DAtarget);
	float kfund_z = 2.*M_PI/(Lz*DAfid/DAtarget);

	switch (los_dir){
		case 0:
			kfund_x = 2.*M_PI/(Lx*Htarget/Hfid);
			break;
		case 1:
			kfund_y = 2.*M_PI/(Ly*Htarget/Hfid);
			break;
		case 2:
			kfund_z = 2.*M_PI/(Lz*Htarget/Hfid);
			break;
		default:
			kfund_z = 2.*M_PI/(Lz*Htarget/Hfid);
			break;
	}
	float knyq_x = kfund_x * nx/2.;
	float knyq_y = kfund_y * ny/2.;
	float knyq_z = kfund_z * nz/2.;
	float knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	float vol(Lx*DAfid/DAtarget*Ly*DAfid/DAtarget*Lz*Htarget/Hfid);
	float shotnoise = vol/(float)ngal;
	//std::cerr << "Fundamental wave numbers: " << kfund_x << "  " << kfund_y << "  " << kfund_z << std::endl;
	//std::cerr << "Number of grids: " << nx << "  " << ny << "  " << nz << std::endl;
	//std::cerr << "Volume: " << vol << std::endl;
	//std::cerr << "Bins: " << nbin << "bins in [" << kmin << ", " << kmax << "), logartithm?" << logbin << std::endl;
	//#pragma omp parallel
	{
		//	int ntasks = omp_get_num_threads();
		//	int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float kx, ky, kz;
		float fre, fim;
		float knorm, pnorm, mu;
		float weight;
		float wx, wy, wz;
		float wx2, wy2, wz2;
		//BinnedData local_pow = powerspec;
		//#pragma omp for
		for(int i=0;i<nx;i++){
			ii = (i>nx/2)?i-(int)nx:i;
			kx = kfund_x * ii;
			if(fabs(kx)>powerspec.get_max()) continue;
			if(fabs(kx)>knyq_x) continue;
			wx = 1.0/gsl_sf_sinc(ii/((float)nx));
			wx *= wx; // NGP
			wx *= wx; // CIC
			wx2 = cos(M_PI * ii/((float)nx));
			wx2 = (1.+2*wx2+wx2*wx2)*(2+wx2)/12.;
			for(int j=0;j<ny;j++){
				jj = (j>ny/2)?j-(int)ny:j;
				ky = kfund_y * jj;
				if(fabs(ky)>powerspec.get_max()) continue;
				if(fabs(ky)>knyq_y) continue;
				wy = 1.0/gsl_sf_sinc(jj/((float)ny));
				wy *= wy; // NGP
				wy *= wy; // CIC
				wy2 = cos(M_PI * jj/((float)ny));
				wy2 = (1.+2*wy2+wy2*wy2)*(2+wy2)/12.;
				for(int k=0;k<(int)nz/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>nz/2)?k-(int)nz:k;
					kz = kfund_z * kk;
					if(fabs(kz)>powerspec.get_max()) continue;
					if(fabs(kz)>knyq_z) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>powerspec.get_max()) continue;
					if(knorm>knyq_min) continue;
					wz = 1.0/gsl_sf_sinc(kk/((float)nz));
					wz *= wz; // NGP
					wz *= wz; // CIC
					wz2 = cos(M_PI * kk/((float)nz));
					wz2 = (1.+2*wz2+wz2*wz2)*(2+wz2)/12.;
					switch (los_dir){
						case 0:
							mu = kx/knorm;
							break;
						case 1:
							mu = ky/knorm;
							break;
						case 2:
							mu = kz/knorm;
							break;
						default:
							mu = kz/knorm;
							break;
					}
					fre = this->get_data_re(i,j,k);
					fim = this->get_data_im(i,j,k);
					pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim)-shotnoise*wx2*wy2*wz2);
					pnorm *= (2.*ell+1.)*gsl_sf_legendre_Pl(ell, mu);
					if(k==0) weight = 0.5;
					else weight = 1.;
					//local_pow.push_data(knorm,pnorm,weight);
					powerspec.push_data(knorm,pnorm,weight);
				}
			}
		}
		//#pragma omp critical
		{
			//powerspec = powerspec+local_pow;
		}
	}
	// return powerspec;
}

#if 0
BinnedData FieldData::calc_power_x(int nbin, float kmin,float kmax,bool logbin,int ell,float Omegam,float Omegam_fid,float redshift) const {
	BinnedData powerspec(nbin,kmin,kmax,logbin);

	//float DAtrue(ZtoComovingD(redshift,Omegam));
    float DAtarget(ZtoComovingD(redshift,Omegam));
	float DAfid(ZtoComovingD(redshift,Omegam_fid));
	//float Htrue(Hz(redshift,Omegam));
	float Htarget(Hz(redshift,Omegam));
	float Hfid(Hz(redshift,Omegam_fid));
	if(redshift < 1e-4) DAtarget = DAfid = 1.;

	float kfund_x = 2.*M_PI/(Lx*Htarget/Hfid);
	float kfund_y = 2.*M_PI/(Ly*DAfid/DAtarget);
	float kfund_z = 2.*M_PI/(Lz*DAfid/DAtarget);
	float knyq_x = kfund_x * nx/2.;
	float knyq_y = kfund_y * ny/2.;
	float knyq_z = kfund_z * nz/2.;
	float knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	float vol(Lx*DAfid/DAtarget*Ly*DAfid/DAtarget*Lz*Htarget/Hfid);
	float shotnoise = vol/(float)ngal;
	//std::cerr << "Fundamental wave numbers: " << kfund_x << "  " << kfund_y << "  " << kfund_z << std::endl;
	//std::cerr << "Number of grids: " << nx << "  " << ny << "  " << nz << std::endl;
	//std::cerr << "Volume: " << vol << std::endl;
	//std::cerr << "Bins: " << nbin << "bins in [" << kmin << ", " << kmax << "), logartithm?" << logbin << std::endl;
	//#pragma omp parallel
	{
		//	int ntasks = omp_get_num_threads();
		//	int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float kx, ky, kz;
		float fre, fim;
		float knorm, pnorm, mu;
		float weight;
		float wx, wy, wz;
		float wx2, wy2, wz2;
		//BinnedData local_pow = powerspec;
		//#pragma omp for
		for(int i=0;i<nz;i++){
			ii = (i>nz/2)?i-(int)nz:i;
			kz = kfund_z * ii;
			if(fabs(kz)>powerspec.get_max()) continue;
			if(fabs(kz)>knyq_z) continue;
			wz = 1.0/gsl_sf_sinc(ii/((float)nz));
			wz *= wz; // NGP
			wz *= wz; // CIC
			//wx2 = sin(M_PI * ii/((float)nx));
			//wx2 = 1. - 2/3. * (wx2*wx2);       // shotnoise factor for CIC assignment
			wz2 = cos(M_PI * ii/((float)nz));
			wz2 = (1.+2*wz2+wz2*wz2)*(2+wz2)/12.;
			for(int j=0;j<ny;j++){
				jj = (j>ny/2)?j-(int)ny:j;
				ky = kfund_y * jj;
				if(fabs(ky)>powerspec.get_max()) continue;
				if(fabs(ky)>knyq_y) continue;
				wy = 1.0/gsl_sf_sinc(jj/((float)ny));
				wy *= wy; // NGP
				wy *= wy; // CIC
				//wy2 = sin(M_PI * jj/((float)ny));
				//wy2 = 1. - 2/3. * (wy2*wy2);       // shotnoise factor for CIC assignment
				wy2 = cos(M_PI * jj/((float)ny));
				wy2 = (1.+2*wy2+wy2*wy2)*(2+wy2)/12.;
				for(int k=0;k<(int)nx/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>nx/2)?k-(int)nx:k;
					kx = kfund_x * kk;
					if(fabs(kx)>powerspec.get_max()) continue;
					if(fabs(kx)>knyq_x) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>powerspec.get_max()) continue;
					if(knorm>knyq_min) continue;
					wx = 1.0/gsl_sf_sinc(kk/((float)nx));
					wx *= wx; // NGP
					wx *= wx; // CIC
					//wz2 = sin(M_PI * kk/((float)nz));
					//wz2 = 1. - 2/3. * (wz2*wz2);       // shotnoise factor for CIC assignment
					wx2 = cos(M_PI * kk/((float)nx));
					wx2 = (1.+2*wx2+wx2*wx2)*(2+wx2)/12.;
					mu = kx/knorm;
					fre = this->get_data_re(i,j,k);
					fim = this->get_data_im(i,j,k);
					pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim)-shotnoise*wx2*wy2*wz2);
					//pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim)-shotnoise);
					pnorm *= (2.*ell+1.)*gsl_sf_legendre_Pl(ell, mu);
					if(k==0) weight = 0.5;
					else weight = 1.;
					//local_pow.push_data(knorm,pnorm,weight);
					powerspec.push_data(knorm,pnorm,weight);
				}
			}
		}
		//#pragma omp critical
		{
			//powerspec = powerspec+local_pow;
		}
	}
	return powerspec;
}

BinnedData FieldData::calc_power_y(int nbin, float kmin,float kmax,bool logbin,int ell,float Omegam,float Omegam_fid,float redshift) const {
	BinnedData powerspec(nbin,kmin,kmax,logbin);

	//float DAtrue(ZtoComovingD(redshift,Omegam));
    float DAtarget(ZtoComovingD(redshift,Omegam));
	float DAfid(ZtoComovingD(redshift,Omegam_fid));
	//float Htrue(Hz(redshift,Omegam));
	float Htarget(Hz(redshift,Omegam));
	float Hfid(Hz(redshift,Omegam_fid));
	if(redshift < 1e-4) DAtarget = DAfid = 1.;

	float kfund_x = 2.*M_PI/(Lx*DAfid/DAtarget);
	float kfund_y = 2.*M_PI/(Ly*Htarget/Hfid);
	float kfund_z = 2.*M_PI/(Lz*DAfid/DAtarget);
	float knyq_x = kfund_x * nx/2.;
	float knyq_y = kfund_y * ny/2.;
	float knyq_z = kfund_z * nz/2.;
	float knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	float vol(Lx*DAfid/DAtarget*Ly*DAfid/DAtarget*Lz*Htarget/Hfid);
	float shotnoise = vol/(float)ngal;
	//std::cerr << "Fundamental wave numbers: " << kfund_x << "  " << kfund_y << "  " << kfund_z << std::endl;
	//std::cerr << "Number of grids: " << nx << "  " << ny << "  " << nz << std::endl;
	//std::cerr << "Volume: " << vol << std::endl;
	//std::cerr << "Bins: " << nbin << "bins in [" << kmin << ", " << kmax << "), logartithm?" << logbin << std::endl;
	//#pragma omp parallel
	{
		//	int ntasks = omp_get_num_threads();
		//	int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float kx, ky, kz;
		float fre, fim;
		float knorm, pnorm, mu;
		float weight;
		float wx, wy, wz;
		float wx2, wy2, wz2;
		//BinnedData local_pow = powerspec;
		//#pragma omp for
		for(int i=0;i<nx;i++){
			ii = (i>nx/2)?i-(int)nx:i;
			kx = kfund_x * ii;
			if(fabs(kx)>powerspec.get_max()) continue;
			if(fabs(kx)>knyq_x) continue;
			wx = 1.0/gsl_sf_sinc(ii/((float)nx));
			wx *= wx; // NGP
			wx *= wx; // CIC
			//wx2 = sin(M_PI * ii/((float)nx));
			//wx2 = 1. - 2/3. * (wx2*wx2);       // shotnoise factor for CIC assignment
			wx2 = cos(M_PI * ii/((float)nx));
			wx2 = (1.+2*wx2+wx2*wx2)*(2+wx2)/12.;
			for(int j=0;j<nz;j++){
				jj = (j>nz/2)?j-(int)nz:j;
				kz = kfund_z * jj;
				if(fabs(kz)>powerspec.get_max()) continue;
				if(fabs(kz)>knyq_z) continue;
				wz = 1.0/gsl_sf_sinc(jj/((float)nz));
				wz *= wz; // NGP
				wz *= wz; // CIC
				//wy2 = sin(M_PI * jj/((float)ny));
				//wy2 = 1. - 2/3. * (wy2*wy2);       // shotnoise factor for CIC assignment
				wz2 = cos(M_PI * jj/((float)nz));
				wz2 = (1.+2*wz2+wz2*wz2)*(2+wz2)/12.;
				for(int k=0;k<(int)ny/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>ny/2)?k-(int)ny:k;
					ky = kfund_y * kk;
					if(fabs(ky)>powerspec.get_max()) continue;
					if(fabs(ky)>knyq_y) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>powerspec.get_max()) continue;
					if(knorm>knyq_min) continue;
					wy = 1.0/gsl_sf_sinc(kk/((float)ny));
					wy *= wy; // NGP
					wy *= wy; // CIC
					//wz2 = sin(M_PI * kk/((float)nz));
					//wz2 = 1. - 2/3. * (wz2*wz2);       // shotnoise factor for CIC assignment
					wy2 = cos(M_PI * kk/((float)ny));
					wy2 = (1.+2*wy2+wy2*wy2)*(2+wy2)/12.;
					mu = ky/knorm;
					fre = this->get_data_re(i,j,k);
					fim = this->get_data_im(i,j,k);
					pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim)-shotnoise*wx2*wy2*wz2);
					//pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim)-shotnoise);
					pnorm *= (2.*ell+1.)*gsl_sf_legendre_Pl(ell, mu);
					if(k==0) weight = 0.5;
					else weight = 1.;
					//local_pow.push_data(knorm,pnorm,weight);
					powerspec.push_data(knorm,pnorm,weight);
				}
			}
		}
		//#pragma omp critical
		{
			//powerspec = powerspec+local_pow;
		}
	}
	return powerspec;
}
#endif

BinnedData FieldData::calc_cross_power_noshot(const FieldData &f1, const FieldData &f2, int nbin, float kmin,float kmax,bool logbin,bool CIC,int ell){
	BinnedData powerspec(nbin,kmin,kmax,logbin);
	int nx = f1.get_nx();
	int ny = f1.get_ny();
	int nz = f1.get_nz();
	double Lx = f1.get_Lx();
	double Ly = f1.get_Ly();
	double Lz = f1.get_Lz();

	float kfund_x = 2.*M_PI/(Lx);
	float kfund_y = 2.*M_PI/(Ly);
	float kfund_z = 2.*M_PI/(Lz);
	float knyq_x = kfund_x * nx/2.;
	float knyq_y = kfund_y * ny/2.;
	float knyq_z = kfund_z * nz/2.;
	float knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	float vol(Lx*Ly*Lz);
	// std::cout << vol << std::endl;
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float kx, ky, kz;
		float f1re, f1im;
		float f2re, f2im;
		float knorm, pnorm, mu;
		float weight;
		float wx, wy, wz;
		BinnedData local_pow = powerspec;
		#pragma omp for
		for(int i=0;i<nx;i++){
			ii = (i>nx/2)?i-(int)nx:i;
			kx = kfund_x * ii;
			if(fabs(kx)>powerspec.get_max()) continue;
			if(fabs(kx)>knyq_x) continue;
			wx = 1.0/gsl_sf_sinc(ii/((float)nx));
			wx *= wx; // NGP
			if(CIC) wx *= wx; // CIC
			for(int j=0;j<ny;j++){
				jj = (j>ny/2)?j-(int)ny:j;
				ky = kfund_y * jj;
				if(fabs(ky)>powerspec.get_max()) continue;
				if(fabs(ky)>knyq_y) continue;
				wy = 1.0/gsl_sf_sinc(jj/((float)ny));
				wy *= wy; // NGP
				if(CIC) wy *= wy; // CIC
				for(int k=0;k<nz/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>nz/2)?k-(int)nz:k;
					kz = kfund_z * kk;
					if(fabs(kz)>powerspec.get_max()) continue;
					if(fabs(kz)>knyq_z) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>powerspec.get_max()) continue;
					if(knorm>knyq_min) continue;
					wz = 1.0/gsl_sf_sinc(kk/((float)nz));
					wz *= wz; // NGP
					if(CIC) wz *= wz; // CIC
					mu = kz/knorm;
					f1re = f1.get_data_re(i,j,k);
					f1im = f1.get_data_im(i,j,k);
					f2re = f2.get_data_re(i,j,k);
					f2im = f2.get_data_im(i,j,k);
					pnorm = wx*wy*wz*(vol*(f1re*f2re+f1im*f2im));
					pnorm *= (2.*ell+1.)*gsl_sf_legendre_Pl(ell, mu);
					if(k==0) weight = 0.5;
					else weight = 1.;
					local_pow.push_data(knorm,pnorm,weight);
					//powerspec.push_data(knorm,pnorm,weight);
				}
			}
		}
		#pragma omp critical
		{
			powerspec = powerspec+local_pow;
		}
	}
	return powerspec;
}

BinnedData FieldData::calc_cross_power_noshot(const FieldData &f1, const FieldData &f2, int nbin, float kmin,float kmax,bool logbin,bool CIC1,bool CIC2,int ell){
	BinnedData powerspec(nbin,kmin,kmax,logbin);
	int nx = f1.get_nx();
	int ny = f1.get_ny();
	int nz = f1.get_nz();
	double Lx = f1.get_Lx();
	double Ly = f1.get_Ly();
	double Lz = f1.get_Lz();

	float kfund_x = 2.*M_PI/(Lx);
	float kfund_y = 2.*M_PI/(Ly);
	float kfund_z = 2.*M_PI/(Lz);
	float knyq_x = kfund_x * nx/2.;
	float knyq_y = kfund_y * ny/2.;
	float knyq_z = kfund_z * nz/2.;
	float knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	float vol(Lx*Ly*Lz);
	// std::cout << vol << std::endl;
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float kx, ky, kz;
		float f1re, f1im;
		float f2re, f2im;
		float knorm, pnorm, mu;
		float weight;
		float wx, wy, wz;
		float wx1, wy1, wz1;
		float wx2, wy2, wz2;
		BinnedData local_pow = powerspec;
		#pragma omp for
		for(int i=0;i<nx;i++){
			ii = (i>nx/2)?i-(int)nx:i;
			kx = kfund_x * ii;
			if(fabs(kx)>powerspec.get_max()) continue;
			if(fabs(kx)>knyq_x) continue;
			wx1 = 1.0/gsl_sf_sinc(ii/((float)nx));
			wx2 = wx1;
			if(CIC1) wx1 *= wx1;
			if(CIC2) wx2 *= wx2;
			wx = wx1 * wx2;
			for(int j=0;j<ny;j++){
				jj = (j>ny/2)?j-(int)ny:j;
				ky = kfund_y * jj;
				if(fabs(ky)>powerspec.get_max()) continue;
				if(fabs(ky)>knyq_y) continue;
				wy1 = 1.0/gsl_sf_sinc(jj/((float)ny));
				wy2 = wy1;
				if(CIC1) wy1 *= wy1;
				if(CIC2) wy2 *= wy2;
				wy = wy1 * wy2;
				for(int k=0;k<nz/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>nz/2)?k-(int)nz:k;
					kz = kfund_z * kk;
					if(fabs(kz)>powerspec.get_max()) continue;
					if(fabs(kz)>knyq_z) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>powerspec.get_max()) continue;
					if(knorm>knyq_min) continue;
					wz1 = 1.0/gsl_sf_sinc(kk/((float)nz));
					wz2 = wz1;
					if(CIC1) wz1 *= wz1;
					if(CIC2) wz2 *= wz2;
					wz = wz1 * wz2; // NGP
					mu = kz/knorm;
					f1re = f1.get_data_re(i,j,k);
					f1im = f1.get_data_im(i,j,k);
					f2re = f2.get_data_re(i,j,k);
					f2im = f2.get_data_im(i,j,k);
					pnorm = wx*wy*wz*(vol*(f1re*f2re+f1im*f2im));
					pnorm *= (2.*ell+1.)*gsl_sf_legendre_Pl(ell, mu);
					if(k==0) weight = 0.5;
					else weight = 1.;
					local_pow.push_data(knorm,pnorm,weight);
					//powerspec.push_data(knorm,pnorm,weight);
				}
			}
		}
		#pragma omp critical
		{
			powerspec = powerspec+local_pow;
		}
	}
	return powerspec;
}

BinnedData FieldData::calc_cross_power(const FieldData &f1, const FieldData &f2, int nbin, float kmin,float kmax,bool logbin,bool CIC,int ell){
	BinnedData powerspec(nbin,kmin,kmax,logbin);
	int nx = f1.get_nx();
	int ny = f1.get_ny();
	int nz = f1.get_nz();
	double Lx = f1.get_Lx();
	double Ly = f1.get_Ly();
	double Lz = f1.get_Lz();

	float kfund_x = 2.*M_PI/(Lx);
	float kfund_y = 2.*M_PI/(Ly);
	float kfund_z = 2.*M_PI/(Lz);
	float knyq_x = kfund_x * nx/2.;
	float knyq_y = kfund_y * ny/2.;
	float knyq_z = kfund_z * nz/2.;
	float knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	float vol(Lx*Ly*Lz);

	int Ngal1(f1.get_ngal());
	int Ngal2(f2.get_ngal());

	double Pshot(vol / double(std::max(Ngal1,Ngal2)));
	std::cout << Pshot << std::endl;
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float kx, ky, kz;
		float f1re, f1im;
		float f2re, f2im;
		float knorm, pnorm, mu;
		float weight;
		float wx, wy, wz;
		BinnedData local_pow = powerspec;
		#pragma omp for
		for(int i=0;i<nx;i++){
			ii = (i>nx/2)?i-(int)nx:i;
			kx = kfund_x * ii;
			if(fabs(kx)>powerspec.get_max()) continue;
			if(fabs(kx)>knyq_x) continue;
			wx = 1.0/gsl_sf_sinc(ii/((float)nx));
			wx *= wx; // NGP
			if(CIC) wx *= wx; // CIC
			for(int j=0;j<ny;j++){
				jj = (j>ny/2)?j-(int)ny:j;
				ky = kfund_y * jj;
				if(fabs(ky)>powerspec.get_max()) continue;
				if(fabs(ky)>knyq_y) continue;
				wy = 1.0/gsl_sf_sinc(jj/((float)ny));
				wy *= wy; // NGP
				if(CIC) wy *= wy; // CIC
				for(int k=0;k<nz/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>nz/2)?k-(int)nz:k;
					kz = kfund_z * kk;
					if(fabs(kz)>powerspec.get_max()) continue;
					if(fabs(kz)>knyq_z) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>powerspec.get_max()) continue;
					if(knorm>knyq_min) continue;
					wz = 1.0/gsl_sf_sinc(kk/((float)nz));
					wz *= wz; // NGP
					if(CIC) wz *= wz; // CIC
					mu = kz/knorm;
					f1re = f1.get_data_re(i,j,k);
					f1im = f1.get_data_im(i,j,k);
					f2re = f2.get_data_re(i,j,k);
					f2im = f2.get_data_im(i,j,k);
					pnorm = wx*wy*wz*(vol*(f1re*f2re+f1im*f2im)-Pshot);
					pnorm *= (2.*ell+1.)*gsl_sf_legendre_Pl(ell, mu);
					if(k==0) weight = 0.5;
					else weight = 1.;
					local_pow.push_data(knorm,pnorm,weight);
					//powerspec.push_data(knorm,pnorm,weight);
				}
			}
		}
		#pragma omp critical
		{
			powerspec = powerspec+local_pow;
		}
	}
	return powerspec;
}

BinnedData FieldData::calc_propagator(const FieldData &f1, const FieldData &f2, int nbin, float kmin,float kmax,bool CIC,bool logbin){

	int nx = f1.get_nx();
	int ny = f1.get_ny();
	int nz = f1.get_nz();
	double Lx = f1.get_Lx();
	double Ly = f1.get_Ly();
	double Lz = f1.get_Lz();

	BinnedData plin(nbin,kmin,kmax,logbin);
	BinnedData pcross(nbin,kmin,kmax,logbin);
	//BinnedData pnl(nbin,kmin,kmax,logbin);
	BinnedData propagator(nbin,kmin,kmax,logbin);

	double kfund_x = 2.*M_PI/Lx;
	double kfund_y = 2.*M_PI/Ly;
	double kfund_z = 2.*M_PI/Lz;
	double knyq_x = kfund_x * nx/2.;
	double knyq_y = kfund_y * ny/2.;
	double knyq_z = kfund_z * nz/2.;
	double knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	double vol(Lx*Ly*Lz);
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		double ii, jj, kk;
		double kx, ky, kz;
		double fre1, fim1;
		double fre2, fim2;
		double knorm, pnorm1, pnorm2, pnorm3;
		double weight;
		double wx, wy, wz;
		BinnedData local_plin = plin;
		BinnedData local_pcross = pcross;
		//BinnedData local_pnl = pnl;
		#pragma omp for
		for(int i=0;i<nx;i++){
			ii = (i>nx/2)?i-nx:i;
			kx = kfund_x * ii;
			if(fabs(kx)>plin.get_max()) continue;
			if(fabs(kx)>knyq_x) continue;
			wx = 1.0/gsl_sf_sinc(ii/((double)nx)); // NGP x no-window
			if(CIC) wx *= wx; // CIC x no-window
			for(int j=0;j<ny;j++){
				jj = (j>ny/2)?j-ny:j;
				ky = kfund_y * jj;
				if(fabs(ky)>plin.get_max()) continue;
				if(fabs(ky)>knyq_y) continue;
				wy = 1.0/gsl_sf_sinc(jj/((double)ny)); // NGP x no-window
				if(CIC) wy *= wy; // CIC x no-window
				for(int k=0;k<nz/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>nz/2)?k-nz:k;
					kz = kfund_z * kk;
					if(fabs(kz)>plin.get_max()) continue;
					if(fabs(kz)>knyq_z) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>plin.get_max()) continue;
					if(knorm>knyq_min) continue;
					wz = 1.0/gsl_sf_sinc(kk/((double)nz)); // NGP x no-window
					if(CIC) wz *= wz; // CIC x no-window
					fre1 = f1.get_data_re(i,j,k);
					fim1 = f1.get_data_im(i,j,k);
					fre2 = f2.get_data_re(i,j,k);
					fim2 = f2.get_data_im(i,j,k);
					pnorm1 = vol*(fre1*fre1+fim1*fim1);
					pnorm2 = wx*wy*wz*vol*(fre1*fre2+fim1*fim2);
					//pnorm3 = wx*wy*wz*wx*wy*wz*vol*(fre2*fre2+fim2*fim2);
					if(k==0) weight = 0.5;
					else weight = 1.;
					local_plin.push_data(knorm,pnorm1,weight);
					local_pcross.push_data(knorm,pnorm2,weight);
					//local_pnl.push_data(knorm,pnorm3,weight);
					//powerspec.push_data(knorm,pnorm,weight);
				}
			}
		}
		#pragma omp critical
		{
			plin = plin+local_plin;
			pcross = pcross+local_pcross;
			//pnl = pnl+local_pnl;
		}
	}

	//plin.dump("test_plin.dat");
	//pcross.dump("test_pcross.dat");
	//pnl.dump("test_pnl.dat");
	for(int n=0;n<nbin;n++){
		int num = plin.get_num(n);
		double xs = plin.get_xsum(n);
		double pl = plin.get_ymean(n);
		double pc = pcross.get_ymean(n);
		double pro = pc/pl;
		propagator.put_num((double)num,n);
		propagator.put_ysum((double)num*pro,n);
		propagator.put_xsum(xs,n);
		// std::cout << xs/(double)num << "  " << pro << "  " << pl << "  " << pc << std::endl;
	}
	return propagator;
}

BinnedData_2D FieldData::calc_power_2D(int nbin, float kmin,float kmax,bool logbin, int nbin_mu,float Omegam,float Omegam_fid,float redshift) const {
	BinnedData_2D powerspec(nbin,kmin,kmax,logbin,nbin_mu,0,1.0001,false);

	float DAtrue(ZtoComovingD_LCDM(redshift,Omegam));
	float DAfid(ZtoComovingD_LCDM(redshift,Omegam_fid));
	float Htrue(Hz_LCDM(redshift,Omegam));
	float Hfid(Hz_LCDM(redshift,Omegam_fid));
	if(redshift < 1e-4) DAtrue = DAfid = 1.;

	float kfund_x = 2.*M_PI/(Lx*DAfid/DAtrue);
	float kfund_y = 2.*M_PI/(Ly*DAfid/DAtrue);
	float kfund_z = 2.*M_PI/(Lz*Htrue/Hfid);
	float knyq_x = kfund_x * nx/2.;
	float knyq_y = kfund_y * ny/2.;
	float knyq_z = kfund_z * nz/2.;
	float knyq_min = std::min({knyq_x,knyq_y,knyq_z});
	float vol(Lx*DAfid/DAtrue*Ly*DAfid/DAtrue*Lz*Htrue/Hfid);
	float shotnoise = vol/(float)ngal;
	//std::cerr << "Fundamental wave numbers: " << kfund_x << "  " << kfund_y << "  " << kfund_z << std::endl;
	//std::cerr << "Number of grids: " << nx << "  " << ny << "  " << nz << std::endl;
	//std::cerr << "Volume: " << vol << std::endl;
	//std::cerr << "Bins: " << nbin << "bins in [" << kmin << ", " << kmax << "), logartithm?" << logbin << std::endl;
	//#pragma omp parallel
	{
		//	int ntasks = omp_get_num_threads();
		//	int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float kx, ky, kz;
		float fre, fim;
		float knorm, pnorm, mu;
		float weight;
		float wx, wy, wz;
		float wx2, wy2, wz2;
		//BinnedData local_pow = powerspec;
		//#pragma omp for
		for(int i=0;i<nx;i++){
			ii = (i>nx/2)?i-(int)nx:i;
			kx = kfund_x * ii;
			if(fabs(kx)>powerspec.get_xmax()) continue;
			if(fabs(kx)>knyq_x) continue;
			wx = 1.0/gsl_sf_sinc(ii/((float)nx));
			wx *= wx; // NGP
			wx *= wx; // CIC
			//wx2 = sin(M_PI * ii/((float)nx));
			//wx2 = 1. - 2/3. * (wx2*wx2);       // shotnoise factor for CIC assignment
			wx2 = cos(M_PI * ii/((float)nx));
			wx2 = (1.+2*wx2+wx2*wx2)*(2+wx2)/12.;
			for(int j=0;j<ny;j++){
				jj = (j>ny/2)?j-(int)ny:j;
				ky = kfund_y * jj;
				if(fabs(ky)>powerspec.get_xmax()) continue;
				if(fabs(ky)>knyq_y) continue;
				wy = 1.0/gsl_sf_sinc(jj/((float)ny));
				wy *= wy; // NGP
				wy *= wy; // CIC
				//wy2 = sin(M_PI * jj/((float)ny));
				//wy2 = 1. - 2/3. * (wy2*wy2);       // shotnoise factor for CIC assignment
				wy2 = cos(M_PI * jj/((float)ny));
				wy2 = (1.+2*wy2+wy2*wy2)*(2+wy2)/12.;
				for(int k=0;k<nz/2;k++){
					if(i==0&&j==0&&k==0) continue;
					kk = (k>nz/2)?k-(int)nz:k;
					kz = kfund_z * kk;
					if(fabs(kz)>powerspec.get_xmax()) continue;
					if(fabs(kz)>knyq_z) continue;
					knorm = sqrt(kx*kx+ky*ky+kz*kz);
					if(knorm>powerspec.get_xmax()) continue;
					if(knorm>knyq_min) continue;
					wz = 1.0/gsl_sf_sinc(kk/((float)nz));
					wz *= wz; // NGP
					wz *= wz; // CIC
					//wz2 = sin(M_PI * kk/((float)nz));
					//wz2 = 1. - 2/3. * (wz2*wz2);       // shotnoise factor for CIC assignment
					wz2 = cos(M_PI * kk/((float)nz));
					wz2 = (1.+2*wz2+wz2*wz2)*(2+wz2)/12.;
					mu = kz/knorm;
					fre = this->get_data_re(i,j,k);
					fim = this->get_data_im(i,j,k);
					pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim)-shotnoise*wx2*wy2*wz2);
					//pnorm = wx*wy*wz*(vol*(fre*fre+fim*fim)-shotnoise);
					//pnorm = wx*wy*wz*vol*(fre*fre+fim*fim);
					//pnorm *= (2.*ell+1.)*gsl_sf_legendre_Pl(ell, mu);
					if(k==0) weight = 0.5;
					else weight = 1.;
					//local_pow.push_data(knorm,pnorm,weight);
					powerspec.push_data(knorm,mu,pnorm,weight);
				}
			}
		}
		//#pragma omp critical
		{
			//powerspec = powerspec+local_pow;
		}
	}
	return powerspec;
}

BinnedData FieldData::calc_correlation(int nbin, float rmin,float rmax,bool logbin) const {
	FieldData tmp_field(*this);
	BinnedData correlation(nbin,rmin,rmax,logbin);
	tmp_field.square();
	tmp_field.do_ifft();
	float rgrid_x = Lx/(float)nx;
	float rgrid_y = Ly/(float)ny;
	float rgrid_z = Lz/(float)nz;
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float rx, ry, rz;
		float rnorm, pnorm;
		float weight;
		BinnedData local_xi = correlation;
		#pragma omp for schedule(guided)
		for(int i=0;i<nx;i++){
			ii = (i>(int)nx/2)?i-(int)nx:i;
			rx = ii * rgrid_x;
			if(fabs(rx)>correlation.get_max()) continue;
			for(int j=0;j<ny;j++){
				jj = (j>(int)ny/2)?j-(int)ny:j;
				ry = jj * rgrid_y;
				if(fabs(ry)>correlation.get_max()) continue;
				for(int k=0;k<(int)nz/2;k++){
					kk = (k>(int)nz/2)?k-(int)nz:k;
					rz = kk * rgrid_z;
					if(fabs(rz)>correlation.get_max()) continue;
					rnorm = sqrt(rx*rx+ry*ry+rz*rz);
					if(k==0) weight = 0.5;
					else weight = 1.;
					local_xi.push_data(rnorm,tmp_field.get_data(i,j,k),weight);
					// local_xi.push_data_CIC(rnorm,tmp_field.get_data(i,j,k),weight);
				}
			}
		}
		#pragma omp critical
		{
			correlation = correlation+local_xi;
		}
	}
	return correlation;
}

BinnedData FieldData::calc_correlation(FieldData &f1, int nbin, float rmin,float rmax,bool logbin){
	//FieldData tmp_field = f1 * f1;
	FieldData tmp_field;
	tmp_field.multiply2fields(f1,f1);
	tmp_field.do_ifft();
	float rgrid_x = tmp_field.get_Lx()/(float)tmp_field.get_nx();
	float rgrid_y = tmp_field.get_Ly()/(float)tmp_field.get_ny();
	float rgrid_z = tmp_field.get_Lz()/(float)tmp_field.get_nz();
	BinnedData correlation(nbin,rmin,rmax,logbin);
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float rx, ry, rz;
		float rnorm, pnorm;
		float weight;
		BinnedData local_xi = correlation;
		#pragma omp for schedule(guided)
		for(int i=0;i<tmp_field.get_nx();i++){
			ii = (i>tmp_field.get_nx()/2)?i-tmp_field.get_nx():i;
			rx = ii * rgrid_x;
			if(fabs(rx)>correlation.get_max()) continue;
			for(int j=0;j<tmp_field.get_ny();j++){
				jj = (j>tmp_field.get_ny()/2)?j-tmp_field.get_ny():j;
				ry = jj * rgrid_y;
				if(fabs(ry)>correlation.get_max()) continue;
				for(int k=0;k<tmp_field.get_nz()/2;k++){
					kk = (k>tmp_field.get_nz()/2)?k-tmp_field.get_nz():k;
					rz = kk * rgrid_z;
					if(fabs(rz)>correlation.get_max()) continue;
					rnorm = sqrt(rx*rx+ry*ry+rz*rz);
					if(k==0) weight = 0.5;
					else weight = 1.;
					local_xi.push_data(rnorm,tmp_field.get_data(i,j,k),weight);
					// local_xi.push_data_CIC(rnorm,tmp_field.get_data(i,j,k),weight);
				}
			}
		}
		#pragma omp critical
		{
			correlation = correlation+local_xi;
		}
	}
	return correlation;
}

BinnedData FieldData::calc_cross_correlation(FieldData &f1, FieldData &f2, int nbin, float rmin,float rmax,bool logbin,bool noDC=false){
	//FieldData tmp_field = f1 * f2;
	FieldData tmp_field;
	tmp_field.multiply2fields(f1,f2,noDC);
	tmp_field.do_ifft();
	float rgrid_x = tmp_field.get_Lx()/(float)tmp_field.get_nx();
	float rgrid_y = tmp_field.get_Ly()/(float)tmp_field.get_ny();
	float rgrid_z = tmp_field.get_Lz()/(float)tmp_field.get_nz();
	BinnedData correlation(nbin,rmin,rmax,logbin);
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float rx, ry, rz;
		float rnorm, pnorm;
		float weight;
		BinnedData local_xi = correlation;
		#pragma omp for schedule(guided)
		for(int i=0;i<tmp_field.get_nx();i++){
			ii = (i>tmp_field.get_nx()/2)?i-tmp_field.get_nx():i;
			rx = ii * rgrid_x;
			if(fabs(rx)>correlation.get_max()) continue;
			for(int j=0;j<tmp_field.get_ny();j++){
				jj = (j>tmp_field.get_ny()/2)?j-tmp_field.get_ny():j;
				ry = jj * rgrid_y;
				if(fabs(ry)>correlation.get_max()) continue;
				for(int k=0;k<tmp_field.get_nz()/2;k++){
					kk = (k>tmp_field.get_nz()/2)?k-tmp_field.get_nz():k;
					rz = kk * rgrid_z;
					if(fabs(rz)>correlation.get_max()) continue;
					rnorm = sqrt(rx*rx+ry*ry+rz*rz);
					if(k==0) weight = 0.5;
					else weight = 1.;
					local_xi.push_data(rnorm,tmp_field.get_data(i,j,k),weight);
					// local_xi.push_data_CIC(rnorm,tmp_field.get_data(i,j,k),weight);
				}
			}
		}
		#pragma omp critical
		{
			correlation = correlation+local_xi;
		}
	}
	std::cerr << "Cross correlation done." << std::endl;
	return correlation;
}

BinnedData FieldData::calc_cross_correlation_dummy(const FieldData &f1, const FieldData &f2, int nbin, float rmin,float rmax,bool logbin,bool noDC=false){
	//FieldData tmp_field = f1 * f2;
	change_space(true);
	multiply2fields_dummy(f1,f2,noDC);
	do_ifft();
	float rgrid_x = Lx/(float)nx;
	float rgrid_y = Ly/(float)ny;
	float rgrid_z = Lz/(float)nz;
	BinnedData correlation(nbin,rmin,rmax,logbin);
	#pragma omp parallel
	{
		int ntasks = omp_get_num_threads();
		int thistask = omp_get_thread_num();
		float ii, jj, kk;
		float rx, ry, rz;
		float rnorm, pnorm;
		float weight;
		BinnedData local_xi = correlation;
		#pragma omp for schedule(guided)
		for(int i=0;i<(int)nx;i++){
			ii = (i>((int)nx)/2)?i-(int)nx:i;
			rx = ii * rgrid_x;
			if(fabs(rx)>correlation.get_max()) continue;
			for(int j=0;j<(int)ny;j++){
				jj = (j>((int)ny)/2)?j-(int)ny:j;
				ry = jj * rgrid_y;
				if(fabs(ry)>correlation.get_max()) continue;
				for(int k=0;k<((int)nz)/2;k++){
					kk = (k>((int)nz)/2)?k-(int)nz:k;
					rz = kk * rgrid_z;
					if(fabs(rz)>correlation.get_max()) continue;
					rnorm = sqrt(rx*rx+ry*ry+rz*rz);
					if(k==0) weight = 0.5;
					else weight = 1.;
					local_xi.push_data(rnorm,get_data(i,j,k),weight);
					// local_xi.push_data_CIC(rnorm,get_data(i,j,k),weight);
				}
			}
		}
		#pragma omp critical
		{
			correlation = correlation+local_xi;
		}
	}
	std::cerr << "Cross correlation done." << std::endl;
	return correlation;
}

void FieldData::zero_pad(int nsub, int i, int j, int k, float val=0){
	#pragma omp parallel for schedule(auto)
	for(int ix=0;ix<nx;ix++){
		int reg_x=ix*nsub/nx;
		if(reg_x != i) continue;
		for(int iy=0;iy<ny;iy++){
			int reg_y=iy*nsub/ny;
			if(reg_y != j) continue;
			for(int iz=0;iz<nz;iz++){
				int reg_z=iz*nsub/nz;
				if(reg_z != k) continue;
				data[(ix*ny+iy)*(2*(nz/2+1))+iz] = val;
				// data[(ix*ny+iy)*(2*(nz/2+1))+iz] = -1;
			}
		}
	}

}

// bool FieldData::fft_init = false;
// fftwf_plan FieldData::forward_plan;
// fftwf_plan FieldData::backward_plan;
