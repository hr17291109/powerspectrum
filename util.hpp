#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <cassert>

typedef unsigned long long int U64;


std::string itos(int n){
	std::stringstream str_stream;
	str_stream << n;
	return str_stream.str();
}

std::string itos2(int n){
	std::stringstream str_stream;
	if(n>=10) str_stream << n;
	else str_stream << "0" << n;
	return str_stream.str();
}

std::string itos3(int n){
	std::stringstream str_stream;
	if(n<10) str_stream << "00" << n;
	else if(n<100) str_stream << "0" << n;
	else str_stream << n;
	return str_stream.str();
}

std::string itos4(int n){
	std::stringstream str_stream;
	if(n<10) str_stream << "000" << n;
	else if(n<100) str_stream << "00" << n;
	else if(n<1000) str_stream << "0" << n;
	else str_stream << n;
	return str_stream.str();
}

std::string itos5(int n){
	std::stringstream str_stream;
	if(n<10) str_stream << "0000" << n;
	else if(n<100) str_stream << "000" << n;
	else if(n<1000) str_stream << "00" << n;
	else if(n<10000) str_stream << "0" << n;
	else str_stream << n;
	return str_stream.str();
}



int fact(int n) { // factorial
	if ((n==0)||(n==1))
	return 1;
	else
	return n*fact(n-1);
}

std::streamsize get_fs(std::string fname){

	std::ifstream ifs(fname.c_str(), std::ios_base::binary);

	std::streamsize size = ifs.seekg(0, std::ios::end).tellg();
	ifs.close();
	return size;
}

bool fexists(std::string fname)
{
  	std::ifstream ifile(fname.c_str());
  	return (bool)ifile;
}

inline double my_Leg2(double mu_sqr){
	return 0.5*(3.*mu_sqr - 1.);
}

inline double my_Leg4(double mu_sqr){
	return 0.125*(35.*mu_sqr*mu_sqr - 30.*mu_sqr + 3.);
}

inline double my_Leg6(double mu_sqr){
	return 0.0625*(231.*mu_sqr*mu_sqr*mu_sqr - 315.*mu_sqr*mu_sqr + 105.*mu_sqr - 5.);
}

