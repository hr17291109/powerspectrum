#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>

#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>


class TransferFunction{
	private:
		std::vector<double> TransferTable, ScaleFactorTable, kWaveTable;
		gsl_spline2d *spl;
		gsl_interp_accel *xacc, *yacc;
		void init(std::string,std::string);
		void read_transfer_table(std::string,std::string);
	public:
		TransferFunction() {}
		TransferFunction(std::string s,std::string t){init(s,t);}
		~TransferFunction() {
			if(spl != nullptr) gsl_spline2d_free(spl);
			if(xacc != nullptr) gsl_interp_accel_free(xacc);
			if(yacc != nullptr) gsl_interp_accel_free(yacc);
		}
		double get_transfer(double k, double z){
			double ascale = 1/(1.+z);
			if(k<1e-10) return 0.;
			else return gsl_spline2d_eval(spl,log(ascale),log(k),xacc,yacc);
		}
};

void TransferFunction::init(std::string infile_dir, std::string type){
	read_transfer_table(infile_dir,type);
	spl = gsl_spline2d_alloc(gsl_interp2d_bicubic,ScaleFactorTable.size(),kWaveTable.size());
	xacc = gsl_interp_accel_alloc();
	yacc = gsl_interp_accel_alloc();
	gsl_spline2d_init(spl,&ScaleFactorTable[0],&kWaveTable[0],&TransferTable[0],ScaleFactorTable.size(),kWaveTable.size());
}
void TransferFunction::read_transfer_table(std::string infile_dir, std::string type){
	double tmp_double;
	std::string fname;
	std::ifstream fin;

	fname = infile_dir + "/scalefac_for_trans.dat";
	fin.open(fname.c_str());
	while(!fin.eof()){
		fin >> tmp_double;
		if(fin.eof()) break;
		ScaleFactorTable.push_back(log(tmp_double));
	}
	fin.close();
	//std::cout << ScaleFactorTable.size() << " entries found in " << fname << std::endl;

	fname = infile_dir + "/kwave_for_trans.dat";
	fin.open(fname.c_str());
	while(!fin.eof()){
		fin >> tmp_double;
		if(fin.eof()) break;
		kWaveTable.push_back(log(tmp_double));
	}
	fin.close();

	//std::cout << kWaveTable.size() << " entries found in " << fname << std::endl;

	if(!type.compare("cb")){
		fname = infile_dir + "/trans_cb.dat";
	}else if(!type.compare("tot")){
		fname = infile_dir + "/trans_tot.dat";
	}else{
		std::cerr << "Illegal transfer function type: " << type << ". Aborting." << std::endl;
		exit(1);
	}
	fin.open(fname.c_str());
	while(!fin.eof()){
		fin >> tmp_double;
		if(fin.eof()) break;
		TransferTable.push_back(tmp_double);
	}
	fin.close();

	//std::cout << TransferTable.size() << " entries found in " << fname << std::endl;

}
