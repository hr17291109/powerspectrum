#ifndef BINNEDDATA_HPP
#define BINNEDDATA_HPP
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>


class BinnedData{
	private :
		int nbin;
		bool logbin;
		double minimum, maximum;
		double ratio, difference, logratio;
		std::vector<double>xbin,xmin,xmax,xcen,ysum,xsum,y2sum,num;
		double get_bin_double(double) const;
	public :
		BinnedData(int,double,double,bool);
		~BinnedData() {clear();}
		BinnedData& operator=(const BinnedData&);
		BinnedData& operator+(const BinnedData&);
		void clear() {ysum.clear();y2sum.clear();num.clear();}
		void clear(int,double,double,bool);
		int get_bin(double) const;
		int push_data_get_bin(double, double, double);
		void push_data(double, double, double);
		void push_data_CIC(double, double, double);
		void push_data(double);
		int push_data_get_bin(double);
		void dump(std::string,bool) const;
		void dump_CIC(std::string,bool) const;
		void dump_minimal(std::string,bool) const;
		void dump_num(std::string) const;
		double get_xmin(int n) const {return xmin[n];}
		double get_xmax(int n) const {return xmax[n];}
		double get_xcen(int n) const {return xcen[n];}
		double get_xbin(int n) const {return xbin[n];}
		double get_xmean(int n) const {return xsum[n]/(double)(num[n]+1e-30);}
		double get_ymean(int n) const {return ysum[n]/(double)(num[n]+1e-30);}
		double get_xsum(int n) const {return xsum[n];}
		double get_ysum(int n) const {return ysum[n];}
		double get_y2sum(int n) const {return y2sum[n];}
		void put_xsum(double xs, int n) {xsum[n] = xs;}
		void put_ysum(double ys, int n) {ysum[n] = ys;}
		void put_y2sum(double y2s, int n) {y2sum[n] = y2s;}
		void put_num(double nn, int n) {num[n] = nn;}
		double get_num(int n) const {return num[n];}
		double get_max() const {return maximum;}
		double get_min() const {return minimum;}
		double get_ratio() const {return ratio;}
		double get_difference() const {return difference;}
		double get_logratio() const {return logratio;}
		int get_nbin() const {return nbin;}
		bool get_logbin() const {return logbin;}
};

BinnedData::BinnedData(int a, double b, double c, bool d){
	nbin = a;
	minimum = b;
	maximum = c;
	logbin = d;
	if(logbin&&(minimum<=0||maximum<=0)){
		std::cerr << "Negative values do not fit in logarithmic binning. Aborting." << std::endl;
		exit(1);
	}
	if(logbin){
		ratio = pow(maximum/minimum,1./(double)nbin);
		logratio = log(ratio);
		for(int n=0;n<nbin+1;n++){
				xbin.push_back(minimum*pow(ratio,n));
		}
		for(int n=0;n<nbin;n++){
			xmin.push_back(minimum*pow(ratio,n));
			xmax.push_back(minimum*pow(ratio,n+1));
			xcen.push_back(minimum*pow(ratio,n+0.5));
		}
	}else{
		difference = (maximum-minimum)/(double)nbin;
		for(int n=0;n<nbin+1;n++){
			xbin.push_back(minimum+difference*n);
		}
	  for(int n=0;n<nbin;n++){
			xmin.push_back(minimum+difference*n);
			xmax.push_back(minimum+difference*(n+1));
			xcen.push_back(minimum+difference*(n+0.5));
		}
	}
	xsum.assign(nbin,0);
	ysum.assign(nbin+1,0);
	y2sum.assign(nbin+1,0);
	num.assign(nbin+1,0);
}

void BinnedData::clear(int a, double b, double c, bool d){
	nbin = a;
	minimum = b;
	maximum = c;
	logbin = d;
	if(logbin&&(minimum<=0||maximum<=0)){
		std::cerr << "Negative values do not fit in logarithmic binning. Aborting." << std::endl;
		exit(1);
	}
	xbin.clear();
	xmin.clear();
	xmax.clear();
	xsum.clear();
	xcen.clear();
	ysum.clear();
	y2sum.clear();
	num.clear();

	if(logbin){
		ratio = pow(maximum/minimum,1./(double)nbin);
		logratio = log(ratio);
		for(int n=0;n<nbin+1;n++){
			xbin.push_back(minimum*pow(ratio,n));
    }
		for(int n=0;n<nbin;n++){
			xmin.push_back(minimum*pow(ratio,n));
			xmax.push_back(minimum*pow(ratio,n+1));
			xcen.push_back(minimum*pow(ratio,n+0.5));
		}
	}else{
		difference = (maximum-minimum)/(double)nbin;
		for(int n=0;n<nbin+1;n++){
      xbin.push_back(minimum+difference*n);
    }
		for(int n=0;n<nbin;n++){
			xmin.push_back(minimum+difference*n);
			xmax.push_back(minimum+difference*(n+1));
			xcen.push_back(minimum+difference*(n+0.5));
		}
	}
	xsum.assign(nbin,0);
	ysum.assign(nbin+1,0);
	y2sum.assign(nbin+1,0);
	num.assign(nbin+1,0);
}

int BinnedData::get_bin(double x) const {
	if(x<minimum||x>=maximum)return -1;
	if(logbin) return floor(log(x/minimum)/logratio);
	else return floor((x-minimum)/difference);
}

double BinnedData::get_bin_double(double x) const {
        if(logbin) return log(x/minimum)/logratio;
        else return (x-minimum)/difference;
}

int BinnedData::push_data_get_bin(double x, double y, double w){
        int bin = get_bin(x);
        if(bin>=0){
                xsum[bin] += w*x;
                ysum[bin] += w*y;
                y2sum[bin] += w*y*y;
                num[bin] += w;
        }
        return bin;
}

void BinnedData::push_data(double x, double y, double w){
	int bin = get_bin(x);
	if(bin>=0){
		xsum[bin] += w*x;
		ysum[bin] += w*y;
		y2sum[bin] += w*y*y;
		num[bin] += w;
	}
}

void BinnedData::push_data_CIC(double x, double y, double w){
        double bin_d = get_bin_double(x);
        int bin_m = floor(bin_d);
        int bin_p = bin_m+1;
        double frac;
        if(!logbin) frac = (x-xbin[0])/difference - floor((x-xbin[0])/difference);
        else frac = (log(x)-log(xbin[0]))/logratio - floor((log(x)-log(xbin[0]))/logratio);
        if(bin_m>=0&&bin_m<nbin+1){
                ysum[bin_m] += (1-frac)*w*y;
                y2sum[bin_m] += (1-frac)*w*y*y;
                num[bin_m] += (1-frac)*w;
        }
        if(bin_p>=0&&bin_p<nbin+1){
                ysum[bin_p] += (frac)*w*y;
                y2sum[bin_p] += (frac)*w*y*y;
                num[bin_p] += (frac)*w;
        }
}

void BinnedData::push_data(double x){ // for simple counting
	int bin = get_bin(x);
	if(bin>=0) num[bin] ++;
}


int BinnedData::push_data_get_bin(double x){ // for simple counting
        int bin = get_bin(x);
        if(bin>=0) num[bin] ++;
        return bin;
}

void BinnedData::dump(std::string outfile,bool app=false) const {
	std::ofstream fout;
	fout.precision(12);
	if(app) fout.open(outfile.c_str(),std::ofstream::app);
	else fout.open(outfile.c_str());
	for(int n=0;n<nbin;n++){
		if(num[n]==0) continue;
		fout << xmin[n] << "  " << xsum[n]/num[n] << "  " << xmax[n] << "  "
			<< ysum[n]/num[n] << "  "
			<< sqrt(y2sum[n]/num[n]-ysum[n]/num[n]*ysum[n]/num[n])/sqrt(fmax(num[n]-1.,1e-30))
			<< "  " << num[n] << std::endl;
	}
	fout.close();
}
void BinnedData::dump_CIC(std::string outfile,bool app=false) const {
        std::ofstream fout;
				fout.precision(12);
        if(app) fout.open(outfile.c_str(),std::ofstream::app);
				else fout.open(outfile.c_str());
        for(int n=0;n<nbin+1;n++){
                if(num[n]==0) continue;
                fout << xbin[n] << "  "
                        << ysum[n]/num[n] << "  "
                        << sqrt(y2sum[n]/num[n]-ysum[n]/num[n]*ysum[n]/num[n])/sqrt(fmax(num[n]-1.,1e-30))
                        << "  " << num[n] << std::endl;
        }
        fout.close();
}
void BinnedData::dump_minimal(std::string outfile,bool app=false) const {
	std::ofstream fout;
	fout.precision(12);
	if(app) fout.open(outfile.c_str(),std::ofstream::app);
	else fout.open(outfile.c_str());
	for(int n=0;n<nbin;n++){
		if(num[n]==0) continue;
		//fout << xsum[n]/num[n] << "  "
		fout << xcen[n] << "  "
			<< ysum[n]/num[n] << "  "
			<< num[n] << std::endl;
	}
	fout.close();
}

void BinnedData::dump_num(std::string outfile) const {
	std::ofstream fout;
	fout.precision(12);
	fout.open(outfile.c_str());
	for(int n=0;n<nbin;n++){
		fout << xcen[n] << "  " << num[n] << std::endl;
	}
	fout.close();
}

BinnedData& BinnedData::operator=(const BinnedData& bd)
{
	nbin = bd.get_nbin();
	logbin = bd.get_logbin();
	minimum = bd.get_min();
	maximum = bd.get_max();
	ratio = bd.get_ratio();
	difference = bd.get_difference();
	logratio = bd.get_logratio();
	for(int n=0;n<nbin;n++){
		xmin[n] = bd.get_xmin(n);
		xmax[n] = bd.get_xmax(n);
		xcen[n] = bd.get_xcen(n);
		xsum[n] = bd.get_xsum(n);
	}
	for(int n=0;n<nbin+1;n++){
		xbin[n] = bd.get_xbin(n);
		ysum[n] = bd.get_ysum(n);
		y2sum[n] = bd.get_y2sum(n);
		num[n] = bd.get_num(n);
	}

    return *this;
}

BinnedData& BinnedData::operator+(const BinnedData& bd){
	if(!((nbin == bd.get_nbin())&&(logbin == bd.get_logbin())&&(minimum == bd.get_min())&&(maximum == bd.get_max()))){
		std::cerr << "Warning. Addition of binned data can be done only when 2 data adopt the same binning." << std::endl;
	}else{
		for(int n=0;n<nbin;n++){
			xsum[n] += bd.get_xsum(n);
		}
		for(int n=0;n<nbin+1;n++){
			ysum[n] += bd.get_ysum(n);
			y2sum[n] += bd.get_y2sum(n);
			num[n] += bd.get_num(n);
		}
	}
	return *this;
}



#endif
