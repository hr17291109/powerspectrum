#ifndef BINNEDDATA_2D_HPP
#define BINNEDDATA_2D_HPP
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>


class BinnedData_2D{
	private :
		int nbin1, nbin2;
		bool logbin1, logbin2;
		double minimum1, maximum1;
		double minimum2, maximum2;
		double ratio1, difference1, logratio1;
		double ratio2, difference2, logratio2;
		std::vector<double>xmin,xmax,xcen,ymin,ymax,ycen,xsum,ysum,zsum,z2sum,num;
		int get_bin1(double) const;		
		int get_bin2(double) const;		
	public :
		BinnedData_2D(int,double,double,bool,int,double,double,bool);
		~BinnedData_2D() {clear();}
		BinnedData_2D& operator=(const BinnedData_2D&);
		BinnedData_2D& operator+(const BinnedData_2D&);
		void clear() {xsum.clear();ysum.clear();zsum.clear();z2sum.clear();num.clear();}
		void clear(int,double,double,bool,int,double,double,bool);
		void push_data(double, double, double, double);
		void push_data(double, double);
		void dump(std::string) const;
		void dump_num(std::string) const;
		double get_xmin(int n) const {return xmin[n];}
		double get_xmax(int n) const {return xmax[n];}
		double get_xcen(int n) const {return xcen[n];}
		double get_ymin(int n) const {return ymin[n];}
		double get_ymax(int n) const {return ymax[n];}
		double get_ycen(int n) const {return ycen[n];}
		double get_xmean(int n) const {return xsum[n]/(double)num[n];}
		double get_ymean(int n) const {return ysum[n]/(double)num[n];}
		double get_xsum(int n) const {return xsum[n];}
		double get_ysum(int n) const {return ysum[n];}
		double get_zsum(int n) const {return zsum[n];}
		double get_z2sum(int n) const {return z2sum[n];}
		double get_num(int n) const {return num[n];}
		double get_xmax() const {return maximum1;}
		double get_xmin() const {return minimum1;}
		double get_ymax() const {return maximum2;}
		double get_ymin() const {return minimum2;}
		double get_xratio() const {return ratio1;}
		double get_yratio() const {return ratio2;}
		double get_xdifference() const {return difference1;}
		double get_ydifference() const {return difference2;}
		double get_xlogratio() const {return logratio1;}
		double get_ylogratio() const {return logratio2;}
		int get_xnbin() const {return nbin1;}
		int get_ynbin() const {return nbin2;}
		bool get_xlogbin() const {return logbin1;}
		bool get_ylogbin() const {return logbin2;}
};

BinnedData_2D::BinnedData_2D(int a, double b, double c, bool d,int a2, double b2, double c2, bool d2){
	nbin1 = a;
	minimum1 = b;
	maximum1 = c;
	logbin1 = d;
	if(logbin1&&(minimum1<=0||maximum1<=0)){
		std::cerr << "Negative values do not fit in logarithmic binning. Aborting." << std::endl;
		exit(1);
	}
	if(logbin1){
		ratio1 = pow(maximum1/minimum1,1./(double)nbin1);
		logratio1 = log(ratio1);
		for(int n=0;n<nbin1;n++){
			xmin.push_back(minimum1*pow(ratio1,n));
			xmax.push_back(minimum1*pow(ratio1,n+1));
			xcen.push_back(minimum1*pow(ratio1,n+0.5));
		}
	}else{
		difference1 = (maximum1-minimum1)/(double)nbin1;
		for(int n=0;n<nbin1;n++){
			xmin.push_back(minimum1+difference1*n);
			xmax.push_back(minimum1+difference1*(n+1));
			xcen.push_back(minimum1+difference1*(n+0.5));
		}		
	}
	nbin2 = a2;
	minimum2 = b2;
	maximum2 = c2;
	logbin2 = d2;
	if(logbin2&&(minimum2<=0||maximum2<=0)){
		std::cerr << "Negative values do not fit in logarithmic binning. Aborting." << std::endl;
		exit(1);
	}
	if(logbin2){
		ratio2 = pow(maximum2/minimum2,1./(double)nbin2);
		logratio2 = log(ratio2);
		for(int n=0;n<nbin2;n++){
			ymin.push_back(minimum2*pow(ratio2,n));
			ymax.push_back(minimum2*pow(ratio2,n+1));
			ycen.push_back(minimum2*pow(ratio2,n+0.5));
		}
	}else{
		difference2 = (maximum2-minimum2)/(double)nbin2;
		for(int n=0;n<nbin2;n++){
			ymin.push_back(minimum2+difference2*n);
			ymax.push_back(minimum2+difference2*(n+1));
			ycen.push_back(minimum2+difference2*(n+0.5));
		}		
	}
	xsum.assign(nbin1*nbin2,0);
	ysum.assign(nbin1*nbin2,0);
	zsum.assign(nbin1*nbin2,0);
	z2sum.assign(nbin1*nbin2,0);
	num.assign(nbin1*nbin2,0);
}

void BinnedData_2D::clear(int a, double b, double c, bool d, int a2, double b2, double c2, bool d2){
	nbin1 = a;
	minimum1 = b;
	maximum1 = c;
	logbin1 = d;
	if(logbin1&&(minimum1<=0||maximum1<=0)){
		std::cerr << "Negative values do not fit in logarithmic binning. Aborting." << std::endl;
		exit(1);
	}
	nbin2 = a2;
	minimum2 = b2;
	maximum2 = c2;
	logbin2 = d2;
	if(logbin2&&(minimum2<=0||maximum2<=0)){
		std::cerr << "Negative values do not fit in logarithmic binning. Aborting." << std::endl;
		exit(1);
	}
	xmin.clear();
	xmax.clear();
	xsum.clear();
	xcen.clear();
	ymin.clear();
	ymax.clear();
	ysum.clear();
	ycen.clear();
	zsum.clear();
	z2sum.clear();
	num.clear();

	if(logbin1){
		ratio1 = pow(maximum1/minimum1,1./(double)nbin1);
		logratio1 = log(ratio1);
		for(int n=0;n<nbin1;n++){
			xmin.push_back(minimum1*pow(ratio1,n));
			xmax.push_back(minimum1*pow(ratio1,n+1));
			xcen.push_back(minimum1*pow(ratio1,n+0.5));
		}
	}else{
		difference1 = (maximum1-minimum1)/(double)nbin1;
		for(int n=0;n<nbin1;n++){
			xmin.push_back(minimum1+difference1*n);
			xmax.push_back(minimum1+difference1*(n+1));
			xcen.push_back(minimum1+difference1*(n+0.5));
		}		
	}
	if(logbin2){
		ratio2 = pow(maximum2/minimum2,1./(double)nbin2);
		logratio2 = log(ratio2);
		for(int n=0;n<nbin2;n++){
			ymin.push_back(minimum2*pow(ratio2,n));
			ymax.push_back(minimum2*pow(ratio2,n+1));
			ycen.push_back(minimum2*pow(ratio2,n+0.5));
		}
	}else{
		difference2 = (maximum2-minimum2)/(double)nbin2;
		for(int n=0;n<nbin2;n++){
			ymin.push_back(minimum2+difference2*n);
			ymax.push_back(minimum2+difference2*(n+1));
			ycen.push_back(minimum2+difference2*(n+0.5));
		}		
	}
	xsum.assign(nbin1*nbin2,0);
	ysum.assign(nbin1*nbin2,0);
	zsum.assign(nbin1*nbin2,0);
	z2sum.assign(nbin1*nbin2,0);
	num.assign(nbin1*nbin2,0);
}

int BinnedData_2D::get_bin1(double x) const {
	if(x<minimum1||x>=maximum1)return -1;
	int bin;
	if(logbin1) return int(log(x/minimum1)/logratio1);
	else return int((x-minimum1)/difference1);
}
int BinnedData_2D::get_bin2(double x) const {
	if(x<minimum2||x>=maximum2)return -1;
	int bin;
	if(logbin2) return int(log(x/minimum2)/logratio2);
	else return int((x-minimum2)/difference2);
}


void BinnedData_2D::push_data(double x, double y, double z, double w){
	int bin1 = get_bin1(x);
	int bin2 = get_bin2(y);
	if(bin1>=0 && bin2>=0){
		xsum[bin1*nbin2+bin2] += w*x;
		ysum[bin1*nbin2+bin2] += w*y;
		zsum[bin1*nbin2+bin2] += w*z;
		z2sum[bin1*nbin2+bin2] += w*z*z;
		num[bin1*nbin2+bin2] += w;
	}
}

void BinnedData_2D::push_data(double x, double y){ // for simple counting
	int bin1 = get_bin1(x);
	int bin2 = get_bin2(y);
	if(bin1>=0 && bin2>=0) num[bin1*nbin2+bin2] ++;
}

void BinnedData_2D::dump(std::string outfile) const {
	std::ofstream fout;
	fout.open(outfile.c_str());
	for(int n=0;n<nbin1;n++){
		for(int m=0;m<nbin2;m++){
			int binnow(n*nbin2+m);
			if(num[binnow]==0) continue;
			fout << xmin[n] << "  " << xsum[binnow]/num[binnow] << "  " << xmax[n] << "  "
			<< ymin[m] << "  " << ysum[binnow]/num[binnow] << "  " << ymax[m] << "  "
			<< zsum[binnow]/num[binnow] << "  " 
			<< sqrt(z2sum[binnow]/num[binnow]-zsum[binnow]/num[binnow]*zsum[binnow]/num[binnow])/sqrt(fmax(num[binnow]-1.,1e-30)) 
			<< "  " << num[binnow] << std::endl;
		}
	}
	fout.close();
}

void BinnedData_2D::dump_num(std::string outfile) const {
	std::ofstream fout;
	fout.open(outfile.c_str());
	for(int n=0;n<nbin1;n++){
		for(int m=0;m<nbin2;m++){
			fout << xcen[n] << "  " << ycen[m] << "  " << num[n*nbin2+m] << std::endl;
		}
	}
	fout.close();
}

BinnedData_2D& BinnedData_2D::operator=(const BinnedData_2D& bd)
{
	nbin1 = bd.get_xnbin();
	logbin1 = bd.get_xlogbin();
	minimum1 = bd.get_xmin();
	maximum1 = bd.get_xmax();
	ratio1 = bd.get_xratio();
	difference1 = bd.get_xdifference();
	logratio1 = bd.get_xlogratio();
	nbin2 = bd.get_ynbin();
	logbin2 = bd.get_ylogbin();
	minimum2 = bd.get_ymin();
	maximum2 = bd.get_ymax();
	ratio2 = bd.get_yratio();
	difference2 = bd.get_ydifference();
	logratio2 = bd.get_ylogratio();
	for(int n=0;n<nbin1;n++){
		xmin[n] = bd.get_xmin(n);
		xmax[n] = bd.get_xmax(n);
		xcen[n] = bd.get_xcen(n);
	}
	for(int n=0;n<nbin2;n++){
		ymin[n] = bd.get_ymin(n);
		ymax[n] = bd.get_ymax(n);
		ycen[n] = bd.get_ycen(n);
	}
	for(int n=0;n<nbin1*nbin2;n++){
		xsum[n] = bd.get_xsum(n);
		ysum[n] = bd.get_ysum(n);
		zsum[n] = bd.get_zsum(n);
		z2sum[n] = bd.get_z2sum(n);
		num[n] = bd.get_num(n);
	}

    return *this;
}

BinnedData_2D& BinnedData_2D::operator+(const BinnedData_2D& bd){
	if(!((nbin1 == bd.get_xnbin())&&(logbin1 == bd.get_xlogbin())&&(minimum1 == bd.get_xmin())&&(maximum1 == bd.get_xmax())&&(nbin2 == bd.get_ynbin())&&(logbin2 == bd.get_ylogbin())&&(minimum2 == bd.get_ymin())&&(maximum2 == bd.get_ymax()))){
		std::cerr << "Warning. Addition of binned data can be done only when 2 data adopt the same binning." << std::endl;
	}else{
		for(int n=0;n<nbin1*nbin2;n++){
			xsum[n] += bd.get_xsum(n);
			ysum[n] += bd.get_ysum(n);
			zsum[n] += bd.get_zsum(n);
			z2sum[n] += bd.get_z2sum(n);
			num[n] += bd.get_num(n);
		}
	}
	return *this;
}



#endif
