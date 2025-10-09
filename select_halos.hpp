#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cassert>
#include <chrono>
#include "util.hpp"
#include <omptl/omptl_algorithm>
#include <omp.h>

const int n_profbins(320);

typedef struct myhalo{
	float vmax;
	float mass;
	float radius;
	float pos[3];
#if ZSPACE
	float vel[3];
#endif
	int cell;
	bool subhalo_flag;
#ifdef DQ1
	bool operator<( const struct myhalo& right ) const {
		return (cell == right.cell) ? (mass < right.mass) : cell < right.cell;
	}
#else
	bool operator<( const struct myhalo& right ) const {
		return (cell == right.cell) ? (vmax < right.vmax) : cell < right.cell;
	}
#endif
} myhalo_str;

typedef struct myhalo_full{
	float vmax;
	float mass;
	float radius;
	float pos[3];
#ifdef ZSPACE
	float vel[3];
#endif
	unsigned short int prof[n_profbins];
	int cell;
	bool subhalo_flag;
#ifdef DQ1
	bool operator<( const struct myhalo_full& right ) const {
		return (cell == right.cell) ? (mass < right.mass) : cell < right.cell;
	}
#else
	bool operator<( const struct myhalo_full& right ) const {
		return (cell == right.cell) ? (vmax < right.vmax) : cell < right.cell;
	}
#endif
} myhalo_full_str;

typedef struct myhosthalo{
	float mass;
	float pos[3];
#ifdef ZSPACE
	float vel[3];
#endif
	bool operator<( const struct myhosthalo& right ) const {
		return mass < right.mass;
	}
	bool operator>( const struct myhosthalo& right ) const {
		return mass > right.mass;
	}

} myhosthalo_str;

typedef struct myhosthalo_full{
	float mass;
	float pos[3];
#ifdef ZSPACE
	float vel[3];
#endif
	unsigned short int prof[n_profbins];
	bool operator<( const struct myhosthalo_full& right ) const {
		return mass < right.mass;
	}
	bool operator>( const struct myhosthalo_full& right ) const {
		return mass > right.mass;
	}

} myhosthalo_full_str;

void load_halo(std::string FileBase, int snapnum, double threshold, double mp, double Box, int Npart1d, int ndom, std::vector<myhosthalo_str> &hoststr, long long int ntake)
{
	double logthresh(log(threshold));

	double rmax(Box/(double)ndom); // The maximum radius up to which the profile is recorded.
	double rmin(rmax/1e4);

	float ratio = pow(rmax/rmin,1./n_profbins);
	std::vector<float>rbins(n_profbins+1);
	std::vector<float>rcens(n_profbins);
	for(int j=0;j<n_profbins+1;j++){
		rbins[j] = rmin * pow(ratio,j);
	}
	for(int j=0;j<n_profbins;j++){
		rcens[j] = rmin * pow(ratio,j+1/2.);
	}


	double search_max(5.); // the maximum distance to which the host-sub relation will be checked in Mpc/h
	int ncell1d(std::ceil(Box/search_max)); // divide the box into ncell1d^3 cells to check the host-sub relation
	int ncell3d(ncell1d*ncell1d*ncell1d);
	float nbar = (Npart1d/Box)*(Npart1d/Box)*(Npart1d/Box);
	std::vector<float>cum_nmeans(n_profbins+1);
	for(int j=0;j<n_profbins+1;j++){
		cum_nmeans[j] = 4.*M_PI/3. * rbins[j] * rbins[j] * rbins[j] * nbar;
	}

	std::string buf;
	std::ifstream fin;
	std::vector<float> vmax;
	std::vector<float> pos;
#ifdef ZSPACE
	std::vector<float> vel;
#endif
	std::vector<float> mass;
	std::vector<float> radius;
	std::vector<unsigned short int> prof;
	std::vector<unsigned int> cum_prof;

	int i(0);
	unsigned long long int nhalos(0);
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_vmax."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int nhalos_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		vmax.resize(nhalos+nhalos_tmp);
		fin.read((char *)&vmax[nhalos],nhalos_tmp*sizeof(float));
		fin.close();
		nhalos += nhalos_tmp;
		i++;
	}
	std::cerr << nhalos << " halos found. Vmax files read.";

	unsigned long long int addr(0);
	i=0;
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_profile."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(unsigned short int);
		prof.resize(addr+n_tmp);
		fin.read((char *)&prof[addr],n_tmp*sizeof(unsigned short int));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Profile files read.";

	addr = 0;
	i=0;
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_pos."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		pos.resize(addr+n_tmp);
		fin.read((char *)&pos[addr],n_tmp*sizeof(float));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Position files read.";


#ifdef ZSPACE
	addr = 0;
	i=0;
	while(1){
#ifdef USE_BULKVEL
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_bulkvel."+ itos(i);
#else
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_corevel."+ itos(i);
#endif
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		vel.resize(addr+n_tmp);
		fin.read((char *)&vel[addr],n_tmp*sizeof(float));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Velocity files read.";
#endif

	cum_prof.resize(nhalos * (n_profbins+1));
	for(unsigned long long int n=0;n<nhalos;n++){
		for(int j=0;j<n_profbins;j++){
			cum_prof[n*(n_profbins+1)+j+1] = cum_prof[n*(n_profbins+1)+j] + (unsigned int)prof[n*n_profbins+j];
		}
	}

	std::cerr << " Cumulative profile constructed.";


	mass.resize(nhalos);
	radius.resize(nhalos);

	std::vector<double> logrbins(n_profbins+1);
	std::vector<double> logdelta(n_profbins+1);
	for(int j=0;j<n_profbins+1;j++){
		logrbins[j] = log(rbins[j]);
	}
	for(unsigned long long int n=0;n<nhalos;n++){
		// if(n%1000000==0) std::cerr << n << "\r" << std::flush;
		unsigned long long int istart = n*(n_profbins+1);
		while(cum_prof[istart]==0)istart++;
		if(istart >= (n+1)*(n_profbins+1)){
			std::cerr << "Error! Something is wrong at " << n << std::endl;
			mass[n] = radius[n] = 0.;
			continue;
		}
		int istart2 = istart%(n_profbins+1);
		if(cum_prof[istart]/cum_nmeans[istart2]<threshold){
			istart ++;
			istart2 ++;
			if(istart2 == n_profbins){
				mass[n] = radius[n] = 0.;
				continue;
			}
		}

		if(istart2 < n_profbins){
			int j(0);
			while(cum_prof[istart+j]/cum_nmeans[istart2+j]>threshold){
				j++;
				if(istart2+j>=n_profbins){
					mass[n] = radius[n] = 0.;
					break;
				}
			}
			double logdens_high = log(cum_prof[istart+j-1]/cum_nmeans[istart2+j-1]);
			double logdens_low = log(cum_prof[istart+j]/cum_nmeans[istart2+j]);
			double frac = (logdens_high-logthresh)/(logdens_high-logdens_low);
			double lograd = (1-frac)*log(rbins[istart2+j-1]) + frac * log(rbins[istart2+j]);
			radius[n] = exp(lograd);
			mass[n] = (float)(4.*M_PI/3.*radius[n]*radius[n]*radius[n]*mp*nbar*threshold);
			//correction
			mass[n] *= 1.+0.55*pow(mass[n]/mp,-0.6);
			radius[n] = pow(mass[n]/(4.*M_PI/3.*mp*nbar*threshold),1/3.);

		}
	}

	std::cerr << " Mass determined.";

	std::vector<myhalo_str> hstr(nhalos);
	for(unsigned long long int n=0;n<nhalos;n++){
		hstr[n].vmax = vmax[n];
		hstr[n].mass = mass[n];
		hstr[n].radius = radius[n];
		for(int j=0;j<3;j++){
			hstr[n].pos[j] = pos[3*n+j];
		}
		int ix((floor)(pos[3*n  ]*ncell1d/Box));
		int iy((floor)(pos[3*n+1]*ncell1d/Box));
		int iz((floor)(pos[3*n+2]*ncell1d/Box));
#ifdef ZSPACE
		for(int j=0;j<3;j++){
			hstr[n].vel[j] = vel[3*n+j];
		}
#endif
		if(ix<0) ix += ncell1d;
		else if(ix>=ncell1d) ix -= ncell1d;
		if(iy<0) iy += ncell1d;
		else if(iy>=ncell1d) iy -= ncell1d;
		if(iz<0) iz += ncell1d;
		else if(iz>=ncell1d) iz -= ncell1d;
		hstr[n].cell = (ix * ncell1d + iy) * ncell1d + iz;
	}

	std::cerr << " Data copied to a structure.";

	omptl::sort(hstr.begin(), hstr.end());

	std::cerr << " Data sorted.";

	std::vector<unsigned long long int>start_addr(ncell3d+1);

	int cell_so_far(0);
	for(unsigned long long int n=0;n<nhalos;n++){
		int cell_now(hstr[n].cell);
		if(cell_so_far!=cell_now){
			while(cell_so_far<cell_now){
				start_addr[++cell_so_far]=n;
			}
		}
	}
	start_addr[ncell3d] = nhalos;

	std::cerr << " Cell address determined.";

	for(unsigned long long int n=0;n<nhalos;n++){
		int cell_now(hstr[n].cell);
		int ix = cell_now/(ncell1d*ncell1d);
		int iy = (cell_now - ix*ncell1d*ncell1d)/ncell1d;
		int iz = cell_now - ix*ncell1d*ncell1d - iy*ncell1d;
		for(int iix=ix-1;iix<=ix+1;iix++){
			if(hstr[n].subhalo_flag) break;
			int iiix = (iix>=0) ? ((iix<ncell1d) ? iix : iix-ncell1d) : iix+ncell1d;
			for(int iiy=iy-1;iiy<=iy+1;iiy++){
				if(hstr[n].subhalo_flag) break;
				int iiiy = (iiy>=0) ? ((iiy<ncell1d) ? iiy : iiy-ncell1d) : iiy+ncell1d;
				for(int iiz=iz-1;iiz<=iz+1;iiz++){
					if(hstr[n].subhalo_flag) break;
					int iiiz = (iiz>=0) ? ((iiz<ncell1d) ? iiz : iiz-ncell1d) : iiz+ncell1d;
					int cell_targ((iiix*ncell1d+iiiy)*ncell1d+iiiz);
					if(cell_now == cell_targ){
						for(unsigned long long int 	m=n+1;m<start_addr[cell_targ+1];m++){
							float dx = hstr[n].pos[0] - hstr[m].pos[0];
							dx = (dx>-Box/2.) ? ((dx<Box/2.) ? dx : dx - Box) : dx + Box;
							float dx2 = dx*dx;
							if(dx2>hstr[m].radius*hstr[m].radius) continue;
							float dy = hstr[n].pos[1] - hstr[m].pos[1];
							dy = (dy>-Box/2.) ? ((dy<Box/2.) ? dy : dy - Box) : dy + Box;
							float dxy2 = dx2+dy*dy;
							if(dxy2>hstr[m].radius*hstr[m].radius) continue;
							float dz = hstr[n].pos[2] - hstr[m].pos[2];
							dz = (dz>-Box/2.) ? ((dz<Box/2.) ? dz : dz - Box) : dz + Box;
							float d2 = dxy2+dz*dz;
							if(d2 < hstr[m].radius*hstr[m].radius){
								hstr[n].subhalo_flag = true;
								break;
							}
						}
					}else{
						for(unsigned long long int 	m=start_addr[cell_targ];m<start_addr[cell_targ+1];m++){
							if(hstr[n].vmax>hstr[m].vmax) continue;
							float dx = hstr[n].pos[0] - hstr[m].pos[0];
							dx = (dx>-Box/2.) ? ((dx<Box/2.) ? dx : dx - Box) : dx + Box;
							float dx2 = dx*dx;
							if(dx2>hstr[m].radius*hstr[m].radius) continue;
							float dy = hstr[n].pos[1] - hstr[m].pos[1];
							dy = (dy>-Box/2.) ? ((dy<Box/2.) ? dy : dy - Box) : dy + Box;
							float dxy2 = dx2+dy*dy;
							if(dxy2>hstr[m].radius*hstr[m].radius) continue;
							float dz = hstr[n].pos[2] - hstr[m].pos[2];
							dz = (dz>-Box/2.) ? ((dz<Box/2.) ? dz : dz - Box) : dz + Box;
							float d2 = dxy2+dz*dz;
							if(d2 < hstr[m].radius*hstr[m].radius){
								hstr[n].subhalo_flag = true;
								break;
							}
						}
					}
				}
			}
		}
	}

	std::cerr << " Hosts/subs determined." << std::endl;

	long long int nhosts(0);
	for(long long int n=0;n<hstr.size();n++){
		if(!hstr[n].subhalo_flag) nhosts++;
	}
	std::cerr << nhosts << " host halos found." << std::endl;
	hoststr.resize(nhosts);
	nhosts=0;
	//int nfake(0);
	for(long long int n=0;n<hstr.size();n++){
		if(!hstr[n].subhalo_flag){
			hoststr[nhosts].mass = hstr[n].mass;
			for(int j=0;j<3;j++){
				hoststr[nhosts].pos[j] = hstr[n].pos[j];
			}
#ifdef ZSPACE
			for(int j=0;j<3;j++){
				hoststr[nhosts].vel[j] = hstr[n].vel[j];
			}
#endif
			nhosts++;
			//if(hstr[n].mass==0) nfake++;
		}
	}
	omptl::sort(hoststr.begin(), hoststr.end(), std::greater<>());


	std::cerr << " Data sorted in descending order of mass." << std::endl;
	if(ntake<nhosts){
		hoststr.resize(ntake);
		std::cerr << ntake << " most massive halos selected." << std::endl;
	}
	std::cerr << "Least massive halo: " << hoststr[ntake-1].mass << " M_sun/h" << std::endl;
}

void load_halo_full_massthreshold(std::string FileBase, int snapnum, double threshold, double mp, double Box, int Npart1d, int ndom, std::vector<myhosthalo_full_str> &hoststr, double Mmin)
{
	double logthresh(log(threshold));

	double rmax(Box/(double)ndom); // The maximum radius up to which the profile is recorded.
	double rmin(rmax/1e4);

	float ratio = pow(rmax/rmin,1./n_profbins);
	std::vector<float>rbins(n_profbins+1);
	std::vector<float>rcens(n_profbins);
	for(int j=0;j<n_profbins+1;j++){
		rbins[j] = rmin * pow(ratio,j);
	}
	for(int j=0;j<n_profbins;j++){
		rcens[j] = rmin * pow(ratio,j+1/2.);
	}


	double search_max(5.); // the maximum distance to which the host-sub relation will be checked in Mpc/h
	int ncell1d(std::ceil(Box/search_max)); // divide the box into ncell1d^3 cells to check the host-sub relation
	int ncell3d(ncell1d*ncell1d*ncell1d);
	float nbar = (Npart1d/Box)*(Npart1d/Box)*(Npart1d/Box);
	std::vector<float>cum_nmeans(n_profbins+1);
	for(int j=0;j<n_profbins+1;j++){
		cum_nmeans[j] = 4.*M_PI/3. * rbins[j] * rbins[j] * rbins[j] * nbar;
	}

	std::string buf;
	std::ifstream fin;
	std::vector<float> vmax;
	std::vector<float> pos;
#ifdef ZSPACE
	std::vector<float> vel;
#endif
	std::vector<float> mass;
	std::vector<float> radius;
	std::vector<unsigned short int> prof;
	std::vector<unsigned int> cum_prof;
#ifdef DQ1
	std::vector<bool> subhalo_flag;
	double fake_cen_factor(1.3);
#endif

	int i(0);
	unsigned long long int nhalos(0);

#ifdef DQ1
	buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_RSM200b.bin";
	fin.open(buf.c_str());
	std::cerr << buf << std::endl;
	nhalos = (unsigned long long int)get_fs(buf)/sizeof(float);
	vmax.resize(nhalos);
	fin.read((char *)&vmax[0],nhalos*sizeof(float));
	fin.close();
	std::cerr << nhalos << " halos found. Rockstar M200b file read.";
#else
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_vmax."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int nhalos_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		vmax.resize(nhalos+nhalos_tmp);
		fin.read((char *)&vmax[nhalos],nhalos_tmp*sizeof(float));
		fin.close();
		nhalos += nhalos_tmp;
		i++;
	}
	std::cerr << nhalos << " halos found. Vmax files read.";
#endif

	unsigned long long int addr(0);
	i=0;
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_profile."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(unsigned short int);
		prof.resize(addr+n_tmp);
		fin.read((char *)&prof[addr],n_tmp*sizeof(unsigned short int));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Profile files read.";

	addr = 0;
	i=0;
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_pos."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		pos.resize(addr+n_tmp);
		fin.read((char *)&pos[addr],n_tmp*sizeof(float));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Position files read.";

#ifdef ZSPACE
	addr = 0;
	i=0;
	while(1){
#ifdef USE_BULKVEL
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_bulkvel."+ itos(i);
#else
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_corevel."+ itos(i);
#endif
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		vel.resize(addr+n_tmp);
		fin.read((char *)&vel[addr],n_tmp*sizeof(float));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Velocity files read.";
#endif

	cum_prof.resize(nhalos * (n_profbins+1));
	for(unsigned long long int n=0;n<nhalos;n++){
		for(int j=0;j<n_profbins;j++){
			cum_prof[n*(n_profbins+1)+j+1] = cum_prof[n*(n_profbins+1)+j] + (unsigned int)prof[n*n_profbins+j];
		}
	}

	std::cerr << " Cumulative profile constructed.";


	mass.resize(nhalos);
	radius.resize(nhalos);

	std::vector<double> logrbins(n_profbins+1);
	std::vector<double> logdelta(n_profbins+1);
	for(int j=0;j<n_profbins+1;j++){
		logrbins[j] = log(rbins[j]);
	}
	for(unsigned long long int n=0;n<nhalos;n++){
		unsigned long long int istart = n*(n_profbins+1);
		while(cum_prof[istart]==0)istart++;
		if(istart >= (n+1)*(n_profbins+1)){
			std::cerr << "Error! Something is wrong at " << n << std::endl;
			mass[n] = radius[n] = 0.;
			continue;
		}
		int istart2 = istart%(n_profbins+1);
		if(cum_prof[istart]/cum_nmeans[istart2]<threshold){
			istart ++;
			istart2 ++;
			if(istart2 == n_profbins){
				mass[n] = radius[n] = 0.;
				continue;
			}
		}

		if(istart2 < n_profbins){
			int j(0);
			while(cum_prof[istart+j]/cum_nmeans[istart2+j]>threshold){
				j++;
				if(istart2+j>=n_profbins){
					mass[n] = radius[n] = 0.;
					break;
				}
			}
			double logdens_high = log(cum_prof[istart+j-1]/cum_nmeans[istart2+j-1]);
			double logdens_low = log(cum_prof[istart+j]/cum_nmeans[istart2+j]);
			double frac = (logdens_high-logthresh)/(logdens_high-logdens_low);
			double lograd = (1-frac)*log(rbins[istart2+j-1]) + frac * log(rbins[istart2+j]);
			radius[n] = exp(lograd);
			mass[n] = (float)(4.*M_PI/3.*radius[n]*radius[n]*radius[n]*mp*nbar*threshold);
			//correction
#ifndef DQ1
			mass[n] *= 1.+0.55*pow(mass[n]/mp,-0.6);
#endif
			radius[n] = pow(mass[n]/(4.*M_PI/3.*mp*nbar*threshold),1/3.);
		}
	}

	std::cerr << " Mass determined.";

#ifdef DQ1
	int nfake(0);
	subhalo_flag.resize(nhalos);
	for(auto n=0;n<nhalos;n++){
		if(mass[n]>fake_cen_factor*vmax[n]){
			subhalo_flag[n] = true;
			nfake ++;
		}else subhalo_flag[n] = false;
	}
	std::cout << nfake << " halos marked as fake." << std::endl;
#endif

	unsigned long long int counter(0);
	std::vector<myhalo_full_str> hstr(nhalos);
	for(unsigned long long int n=0;n<nhalos;n++){
#ifdef DQ1
		if(subhalo_flag[n]) continue;
#endif
		hstr[counter].vmax = vmax[n];
		hstr[counter].mass = mass[n];
		hstr[counter].radius = radius[n];
		for(int j=0;j<3;j++){
			hstr[counter].pos[j] = pos[3*n+j];
		}
#ifdef ZSPACE
		for(int j=0;j<3;j++){
			hstr[counter].vel[j] = vel[3*n+j];
		}
#endif
		for(int j=0;j<n_profbins;j++){
			hstr[counter].prof[j] = prof[(unsigned long long int)n_profbins*n+j];
		}
		int ix((floor)(pos[3*n  ]*ncell1d/Box));
		int iy((floor)(pos[3*n+1]*ncell1d/Box));
		int iz((floor)(pos[3*n+2]*ncell1d/Box));
		if(ix<0) ix += ncell1d;
		else if(ix>=ncell1d) ix -= ncell1d;
		if(iy<0) iy += ncell1d;
		else if(iy>=ncell1d) iy -= ncell1d;
		if(iz<0) iz += ncell1d;
		else if(iz>=ncell1d) iz -= ncell1d;
		hstr[counter++].cell = (ix * ncell1d + iy) * ncell1d + iz;
	}

#ifdef DQ1
	hstr.resize(counter);
	nhalos = counter;
	std::cerr << nhalos << " halos survive." << std::endl;
#endif
	std::cerr << " Data copied to a structure.";

	omptl::sort(hstr.begin(), hstr.end());

	std::cerr << " Data sorted.";

	std::vector<unsigned long long int>start_addr(ncell3d+1);

	int cell_so_far(0);
	for(unsigned long long int n=0;n<nhalos;n++){
		int cell_now(hstr[n].cell);
		if(cell_so_far!=cell_now){
			while(cell_so_far<cell_now){
				start_addr[++cell_so_far]=n;
			}
		}
	}
	start_addr[ncell3d] = nhalos;

	std::cerr << " Cell address determined.";

	for(unsigned long long int n=0;n<nhalos;n++){
		int cell_now(hstr[n].cell);
		int ix = cell_now/(ncell1d*ncell1d);
		int iy = (cell_now - ix*ncell1d*ncell1d)/ncell1d;
		int iz = cell_now - ix*ncell1d*ncell1d - iy*ncell1d;
		for(int iix=ix-1;iix<=ix+1;iix++){
			if(hstr[n].subhalo_flag) break;
			int iiix = (iix>=0) ? ((iix<ncell1d) ? iix : iix-ncell1d) : iix+ncell1d;
			for(int iiy=iy-1;iiy<=iy+1;iiy++){
				if(hstr[n].subhalo_flag) break;
				int iiiy = (iiy>=0) ? ((iiy<ncell1d) ? iiy : iiy-ncell1d) : iiy+ncell1d;
				for(int iiz=iz-1;iiz<=iz+1;iiz++){
					if(hstr[n].subhalo_flag) break;
					int iiiz = (iiz>=0) ? ((iiz<ncell1d) ? iiz : iiz-ncell1d) : iiz+ncell1d;
					int cell_targ((iiix*ncell1d+iiiy)*ncell1d+iiiz);
					if(cell_now == cell_targ){
						for(unsigned long long int 	m=n+1;m<start_addr[cell_targ+1];m++){
							float dx = hstr[n].pos[0] - hstr[m].pos[0];
							dx = (dx>-Box/2.) ? ((dx<Box/2.) ? dx : dx - Box) : dx + Box;
							float dx2 = dx*dx;
							if(dx2>hstr[m].radius*hstr[m].radius) continue;
							float dy = hstr[n].pos[1] - hstr[m].pos[1];
							dy = (dy>-Box/2.) ? ((dy<Box/2.) ? dy : dy - Box) : dy + Box;
							float dxy2 = dx2+dy*dy;
							if(dxy2>hstr[m].radius*hstr[m].radius) continue;
							float dz = hstr[n].pos[2] - hstr[m].pos[2];
							dz = (dz>-Box/2.) ? ((dz<Box/2.) ? dz : dz - Box) : dz + Box;
							float d2 = dxy2+dz*dz;
							if(d2 < hstr[m].radius*hstr[m].radius){
								hstr[n].subhalo_flag = true;
								break;
							}
						}
					}else{
						for(unsigned long long int 	m=start_addr[cell_targ];m<start_addr[cell_targ+1];m++){
#ifdef DQ1
							if(hstr[n].mass>hstr[m].mass) continue;
#else
							if(hstr[n].vmax>hstr[m].vmax) continue;
#endif
							float dx = hstr[n].pos[0] - hstr[m].pos[0];
							dx = (dx>-Box/2.) ? ((dx<Box/2.) ? dx : dx - Box) : dx + Box;
							float dx2 = dx*dx;
							if(dx2>hstr[m].radius*hstr[m].radius) continue;
							float dy = hstr[n].pos[1] - hstr[m].pos[1];
							dy = (dy>-Box/2.) ? ((dy<Box/2.) ? dy : dy - Box) : dy + Box;
							float dxy2 = dx2+dy*dy;
							if(dxy2>hstr[m].radius*hstr[m].radius) continue;
							float dz = hstr[n].pos[2] - hstr[m].pos[2];
							dz = (dz>-Box/2.) ? ((dz<Box/2.) ? dz : dz - Box) : dz + Box;
							float d2 = dxy2+dz*dz;
							if(d2 < hstr[m].radius*hstr[m].radius){
								hstr[n].subhalo_flag = true;
								break;
							}
						}
					}
				}
			}
		}
	}

	std::cerr << " Hosts/subs determined." << std::endl;

	long long int nhosts(0);
	for(long long int n=0;n<hstr.size();n++){
		if(!hstr[n].subhalo_flag && hstr[n].mass >= Mmin) nhosts++;
	}
	std::cerr << nhosts << " host halos found in the interested mass range." << std::endl;
	hoststr.resize(nhosts);
	nhosts=0;
	//int nfake(0);
	for(long long int n=0;n<hstr.size();n++){
		if(!hstr[n].subhalo_flag && hstr[n].mass >= Mmin){
			hoststr[nhosts].mass = hstr[n].mass;
			for(int j=0;j<3;j++){
				hoststr[nhosts].pos[j] = hstr[n].pos[j];
			}
#ifdef ZSPACE
			for(int j=0;j<3;j++){
				hoststr[nhosts].vel[j] = hstr[n].vel[j];
			}
#endif
			for(int i=0;i<n_profbins;i++){
				hoststr[nhosts].prof[i] = hstr[n].prof[i];
			}
			nhosts++;
			//if(hstr[n].mass==0) nfake++;
		}
	}
	omptl::sort(hoststr.begin(), hoststr.end(), std::greater<>());


	std::cerr << "Data sorted in descending order of mass." << std::endl;
	std::cerr << "Least massive halo: " << hoststr[nhosts-1].mass << " M_sun/h" << std::endl;
	std::cerr << "Most massive halo: " << hoststr[0].mass << " M_sun/h" << std::endl;
}


void load_halo_full_vmaxthreshold(std::string FileBase, int snapnum, std::vector<myhosthalo_full_str> &hoststr)
{
	std::string buf;
	std::ifstream fin;
	std::vector<float> pos;
#ifdef ZSPACE
	std::vector<float> vel;
#endif
	std::vector<float> mass; // Use this array to store vmax.
	std::vector<unsigned short int> prof;

	int i(0);
	unsigned long long int nhalos(0);

	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_vmax."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int nhalos_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		mass.resize(nhalos+nhalos_tmp);
		fin.read((char *)&mass[nhalos],nhalos_tmp*sizeof(float));
		fin.close();
		nhalos += nhalos_tmp;
		i++;
	}
	std::cerr << nhalos << " halos found. Vmax files read.";

	unsigned long long int addr(0);
	i=0;
	/*
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_profile."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(unsigned short int);
		prof.resize(addr+n_tmp);
		fin.read((char *)&prof[addr],n_tmp*sizeof(unsigned short int));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Profile files read.";
	*/

	addr = 0;
	i=0;
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_pos."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		pos.resize(addr+n_tmp);
		fin.read((char *)&pos[addr],n_tmp*sizeof(float));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Position files read.";

#ifdef ZSPACE
	addr = 0;
	i=0;
	while(1){
#ifdef USE_BULKVEL
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_bulkvel."+ itos(i);
#else
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_corevel."+ itos(i);
#endif
		fin.open(buf.c_str());
		if (fin.fail()) break;
		std::cerr << buf << std::endl;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		vel.resize(addr+n_tmp);
		fin.read((char *)&vel[addr],n_tmp*sizeof(float));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Velocity files read.";
#endif

	hoststr.resize(nhalos);
	//int nfake(0);
	for(long long int n=0;n<hoststr.size();n++){
		hoststr[n].mass = mass[n];
		for(int j=0;j<3;j++){
			hoststr[n].pos[j] = pos[3*n+j];
		}
#ifdef ZSPACE
		for(int j=0;j<3;j++){
			hoststr[n].vel[j] = vel[3*n+j];
		}
#endif
		// for(int i=0;i<n_profbins;i++){
		// 	hoststr[n].prof[i] = prof[(unsigned long long int)n_profbins*n+i];
		// }
	}
	omptl::sort(hoststr.begin(), hoststr.end(), std::greater<>());


	std::cerr << "Data sorted in descending order of Vmax." << std::endl;
	std::cerr << "Smallest Vmax: " << hoststr[nhalos-1].mass << " km/s" << std::endl;
	std::cerr << "Largest Vmax: " << hoststr[0].mass << " km/s" << std::endl;
}

void stack_halo(std::string FileBase, int snapnum, double threshold, double mp, float Box, int Npart1d, int ndom, std::vector<myhosthalo_full_str> &hoststr, long long int ntake1, long long int ntake2, std::vector<double>& r_direct, std::vector<double>& rmin_direct, std::vector<double>& rmax_direct, std::vector<double> &xihm)
{
	double logthresh(log(threshold));
	double rmax(Box/(double)ndom); // The maximum radius up to which the profile is recorded.
	double rmin(rmax/1e4);

	double ratio = pow(rmax/rmin,1./n_profbins);
	std::vector<double>rbins(n_profbins+1);
	std::vector<double>rcens(n_profbins);
	for(int j=0;j<n_profbins+1;j++){
		rbins[j] = rmin * pow(ratio,j);
	}
	for(int j=0;j<n_profbins;j++){
		rcens[j] = rmin * pow(ratio,j+1/2.);
	}


	double search_max(5.); // the maximum distance to which the host-sub relation will be checked in Mpc/h
	int ncell1d(std::ceil(Box/search_max)); // divide the box into ncell1d^3 cells to check the host-sub relation
	int ncell3d(ncell1d*ncell1d*ncell1d);
	double nbar = (Npart1d/Box)*(Npart1d/Box)*(Npart1d/Box);
	std::vector<double>cum_nmeans(n_profbins+1);
	for(int j=0;j<n_profbins+1;j++){
		cum_nmeans[j] = 4.*M_PI/3. * rbins[j] * rbins[j] * rbins[j] * nbar;
	}

	std::string buf;
	std::ifstream fin;
	std::vector<float> vmax;
	std::vector<float> pos;
	std::vector<float> mass;
	std::vector<float> radius;
	std::vector<unsigned short int> prof;
	std::vector<unsigned int> cum_prof;

	int i(0);
	unsigned long long int nhalos(0);
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_vmax."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		unsigned long long int nhalos_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		vmax.resize(nhalos+nhalos_tmp);
		fin.read((char *)&vmax[nhalos],nhalos_tmp*sizeof(float));
		fin.close();
		nhalos += nhalos_tmp;
		i++;
	}
	std::cerr << nhalos << " halos found. Vmax files read.";

	unsigned long long int addr(0);
	i=0;
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_profile."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(unsigned short int);
		prof.resize(addr+n_tmp);
		fin.read((char *)&prof[addr],n_tmp*sizeof(unsigned short int));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Profile files read.";

	addr = 0;
	i=0;
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_pos."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		pos.resize(addr+n_tmp);
		fin.read((char *)&pos[addr],n_tmp*sizeof(float));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Position files read.";

	/*
	for(unsigned long long int n=0;n<nhalos;n++){
	if(vmax[n]>1000){
	std::cout << vmax[n] << " " << pos[3*n] << " " << pos[3*n+1] << " " << pos[3*n+2] << std::endl;
}
}
*/

cum_prof.resize(nhalos * (n_profbins+1));
for(unsigned long long int n=0;n<nhalos;n++){
	for(int j=0;j<n_profbins;j++){
		cum_prof[n*(n_profbins+1)+j+1] = cum_prof[n*(n_profbins+1)+j] + (unsigned int)prof[n*n_profbins+j];
	}
}

std::cerr << " Cumulative profile constructed.";


mass.resize(nhalos);
radius.resize(nhalos);

std::vector<double> logrbins(n_profbins+1);
std::vector<double> logdelta(n_profbins+1);
for(int j=0;j<n_profbins+1;j++){
	logrbins[j] = log(rbins[j]);
}
for(unsigned long long int n=0;n<nhalos;n++){
	// if(n%1000000==0) std::cerr << n << "\r" << std::flush;
	unsigned long long int istart = n*(n_profbins+1);
	while(cum_prof[istart]==0)istart++;
	if(istart >= (n+1)*(n_profbins+1)){
		std::cerr << "Error! Something is wrong at " << n << std::endl;
		mass[n] = radius[n] = 0.;
		continue;
	}
	int istart2 = istart%(n_profbins+1);
	if(cum_prof[istart]/cum_nmeans[istart2]<threshold){
		istart ++;
		istart2 ++;
		if(istart2 == n_profbins){
			mass[n] = radius[n] = 0.;
			continue;
		}
	}

	if(istart2 < n_profbins){
		int j(0);
		while(cum_prof[istart+j]/cum_nmeans[istart2+j]>threshold){
			j++;
			if(istart2+j>=n_profbins){
				mass[n] = radius[n] = 0.;
				break;
			}
		}
		double logdens_high = log(cum_prof[istart+j-1]/cum_nmeans[istart2+j-1]);
		double logdens_low = log(cum_prof[istart+j]/cum_nmeans[istart2+j]);
		double frac = (logdens_high-logthresh)/(logdens_high-logdens_low);
		double lograd = (1-frac)*log(rbins[istart2+j-1]) + frac * log(rbins[istart2+j]);
		radius[n] = exp(lograd);
		mass[n] = (float)(4.*M_PI/3.*radius[n]*radius[n]*radius[n]*mp*nbar*threshold);
		//correction
		mass[n] *= 1.+0.55*pow(mass[n]/mp,-0.6);
		radius[n] = pow(mass[n]/(4.*M_PI/3.*mp*nbar*threshold),1/3.);
	}
}

std::cerr << " Mass determined.";

std::vector<myhalo_full_str> hstr(nhalos);
for(unsigned long long int n=0;n<nhalos;n++){
	hstr[n].vmax = vmax[n];
	hstr[n].mass = mass[n];
	hstr[n].radius = radius[n];
	for(int j=0;j<3;j++){
		hstr[n].pos[j] = pos[3*n+j];
	}
	for(int j=0;j<n_profbins;j++){
		hstr[n].prof[j] = prof[(unsigned long long int)n_profbins*n+j];
	}
	int ix((floor)(pos[3*n  ]*ncell1d/Box));
	int iy((floor)(pos[3*n+1]*ncell1d/Box));
	int iz((floor)(pos[3*n+2]*ncell1d/Box));
	if(ix<0) ix += ncell1d;
	else if(ix>=ncell1d) ix -= ncell1d;
	if(iy<0) iy += ncell1d;
	else if(iy>=ncell1d) iy -= ncell1d;
	if(iz<0) iz += ncell1d;
	else if(iz>=ncell1d) iz -= ncell1d;
	hstr[n].cell = (ix * ncell1d + iy) * ncell1d + iz;
}

std::cerr << " Data copied to a structure.";

omptl::sort(hstr.begin(), hstr.end());

std::cerr << " Data sorted.";

std::vector<unsigned long long int>start_addr(ncell3d+1);

int cell_so_far(0);
for(unsigned long long int n=0;n<nhalos;n++){
	int cell_now(hstr[n].cell);
	if(cell_so_far!=cell_now){
		while(cell_so_far<cell_now){
			start_addr[++cell_so_far]=n;
		}
	}
}
start_addr[ncell3d] = nhalos;

std::cerr << " Cell address determined.";

for(unsigned long long int n=0;n<nhalos;n++){
	int cell_now(hstr[n].cell);
	int ix = cell_now/(ncell1d*ncell1d);
	int iy = (cell_now - ix*ncell1d*ncell1d)/ncell1d;
	int iz = cell_now - ix*ncell1d*ncell1d - iy*ncell1d;
	for(int iix=ix-1;iix<=ix+1;iix++){
		if(hstr[n].subhalo_flag) break;
		int iiix = (iix>=0) ? ((iix<ncell1d) ? iix : iix-ncell1d) : iix+ncell1d;
		for(int iiy=iy-1;iiy<=iy+1;iiy++){
			if(hstr[n].subhalo_flag) break;
			int iiiy = (iiy>=0) ? ((iiy<ncell1d) ? iiy : iiy-ncell1d) : iiy+ncell1d;
			for(int iiz=iz-1;iiz<=iz+1;iiz++){
				if(hstr[n].subhalo_flag) break;
				int iiiz = (iiz>=0) ? ((iiz<ncell1d) ? iiz : iiz-ncell1d) : iiz+ncell1d;
				int cell_targ((iiix*ncell1d+iiiy)*ncell1d+iiiz);
				if(cell_now == cell_targ){
					for(unsigned long long int 	m=n+1;m<start_addr[cell_targ+1];m++){
						float dx = hstr[n].pos[0] - hstr[m].pos[0];
						dx = (dx>-Box/2.) ? ((dx<Box/2.) ? dx : dx - Box) : dx + Box;
						float dx2 = dx*dx;
						if(dx2>hstr[m].radius*hstr[m].radius) continue;
						float dy = hstr[n].pos[1] - hstr[m].pos[1];
						dy = (dy>-Box/2.) ? ((dy<Box/2.) ? dy : dy - Box) : dy + Box;
						float dxy2 = dx2+dy*dy;
						if(dxy2>hstr[m].radius*hstr[m].radius) continue;
						float dz = hstr[n].pos[2] - hstr[m].pos[2];
						dz = (dz>-Box/2.) ? ((dz<Box/2.) ? dz : dz - Box) : dz + Box;
						float d2 = dxy2+dz*dz;
						if(d2 < hstr[m].radius*hstr[m].radius){
							hstr[n].subhalo_flag = true;
							break;
						}
					}
				}else{
					for(unsigned long long int 	m=start_addr[cell_targ];m<start_addr[cell_targ+1];m++){
						if(hstr[n].vmax>hstr[m].vmax) continue;
						float dx = hstr[n].pos[0] - hstr[m].pos[0];
						dx = (dx>-Box/2.) ? ((dx<Box/2.) ? dx : dx - Box) : dx + Box;
						float dx2 = dx*dx;
						if(dx2>hstr[m].radius*hstr[m].radius) continue;
						float dy = hstr[n].pos[1] - hstr[m].pos[1];
						dy = (dy>-Box/2.) ? ((dy<Box/2.) ? dy : dy - Box) : dy + Box;
						float dxy2 = dx2+dy*dy;
						if(dxy2>hstr[m].radius*hstr[m].radius) continue;
						float dz = hstr[n].pos[2] - hstr[m].pos[2];
						dz = (dz>-Box/2.) ? ((dz<Box/2.) ? dz : dz - Box) : dz + Box;
						float d2 = dxy2+dz*dz;
						if(d2 < hstr[m].radius*hstr[m].radius){
							hstr[n].subhalo_flag = true;
							break;
						}
					}
				}
			}
		}
	}
}

std::cerr << " Hosts/subs determined." << std::endl;


long long int nhosts(0);
for(long long int n=0;n<hstr.size();n++){
	if(!hstr[n].subhalo_flag) nhosts++;
}
std::cerr << nhosts << " host halos found." << std::endl;
hoststr.resize(nhosts);
nhosts=0;
//int nfake(0);
for(long long int n=0;n<hstr.size();n++){
	if(!hstr[n].subhalo_flag){
		hoststr[nhosts].mass = hstr[n].mass;
		for(int j=0;j<3;j++){
			hoststr[nhosts].pos[j] = hstr[n].pos[j];
		}
		for(int i=0;i<n_profbins;i++){
			hoststr[nhosts].prof[i] = hstr[n].prof[i];
		}
		nhosts++;
		//if(hstr[n].mass==0) nfake++;
	}
}
omptl::sort(hoststr.begin(), hoststr.end(), std::greater<>());


std::cerr << " Data sorted in descending order of mass." << std::endl;
if(ntake2<nhosts){
	hoststr.resize(ntake2);
	std::cerr << ntake2 << " most massive halos selected." << std::endl;
}
std::cerr << "Least massive halo: " << hoststr[ntake2-1].mass << " M_sun/h" << std::endl;

xihm.resize(n_profbins);
r_direct.resize(n_profbins);
rmin_direct.resize(n_profbins);
rmax_direct.resize(n_profbins);
for(long long int n=ntake1;n<hoststr.size();n++){
	for(int i=0;i<n_profbins;i++){
		xihm[i] += (double)hoststr[n].prof[i];
	}
}
for(int i=0;i<n_profbins;i++){
	r_direct[i] = rcens[i];
	rmin_direct[i] = rbins[i];
	rmax_direct[i] = rbins[i+1];
	xihm[i] = xihm[i] / (4.*M_PI/3. * (rbins[i+1] * rbins[i+1] * rbins[i+1] - rbins[i] * rbins[i] * rbins[i]) * nbar * (ntake2-ntake1)) - 1.;
}

for(int i=0;i<n_profbins;i++){
	std::cout << rcens[i] << "  " << xihm[i] << std::endl;
}

}

void stack_halo(std::string FileBase, int snapnum, double threshold, double mp, float Box, int Npart1d, int ndom, std::vector<myhosthalo_full_str> &hoststr, double Mmin, double Mmax, std::vector<double>& r_direct, std::vector<double>& rmin_direct, std::vector<double>& rmax_direct, std::vector<double> &xihm)
{
	double logthresh(log(threshold));
	double rmax(Box/(double)ndom); // The maximum radius up to which the profile is recorded.
	double rmin(rmax/1e4);

	double ratio = pow(rmax/rmin,1./n_profbins);
	std::vector<double>rbins(n_profbins+1);
	std::vector<double>rcens(n_profbins);
	for(int j=0;j<n_profbins+1;j++){
		rbins[j] = rmin * pow(ratio,j);
	}
	for(int j=0;j<n_profbins;j++){
		rcens[j] = rmin * pow(ratio,j+1/2.);
	}


	double search_max(5.); // the maximum distance to which the host-sub relation will be checked in Mpc/h
	int ncell1d(std::ceil(Box/search_max)); // divide the box into ncell1d^3 cells to check the host-sub relation
	int ncell3d(ncell1d*ncell1d*ncell1d);
	double nbar = (Npart1d/Box)*(Npart1d/Box)*(Npart1d/Box);
	std::vector<double>cum_nmeans(n_profbins+1);
	for(int j=0;j<n_profbins+1;j++){
		cum_nmeans[j] = 4.*M_PI/3. * rbins[j] * rbins[j] * rbins[j] * nbar;
	}

	std::string buf;
	std::ifstream fin;
	std::vector<float> vmax;
	std::vector<float> pos;
	std::vector<float> mass;
	std::vector<float> radius;
	std::vector<unsigned short int> prof;
	std::vector<unsigned int> cum_prof;

	int i(0);
	unsigned long long int nhalos(0);
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_vmax."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		unsigned long long int nhalos_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		vmax.resize(nhalos+nhalos_tmp);
		fin.read((char *)&vmax[nhalos],nhalos_tmp*sizeof(float));
		fin.close();
		nhalos += nhalos_tmp;
		i++;
	}
	std::cerr << nhalos << " halos found. Vmax files read.";

	unsigned long long int addr(0);
	i=0;
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_profile."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(unsigned short int);
		prof.resize(addr+n_tmp);
		fin.read((char *)&prof[addr],n_tmp*sizeof(unsigned short int));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Profile files read.";

	addr = 0;
	i=0;
	while(1){
		buf = FileBase + "/halo_catalog/S" + itos3(snapnum) + "_pos."+ itos(i);
		fin.open(buf.c_str());
		if (fin.fail()) break;
		unsigned long long int n_tmp = (unsigned long long int)get_fs(buf)/sizeof(float);
		pos.resize(addr+n_tmp);
		fin.read((char *)&pos[addr],n_tmp*sizeof(float));
		fin.close();
		addr += n_tmp;
		i++;
	}
	std::cerr << " Position files read.";

	/*
	for(unsigned long long int n=0;n<nhalos;n++){
	if(vmax[n]>1000){
	std::cout << vmax[n] << " " << pos[3*n] << " " << pos[3*n+1] << " " << pos[3*n+2] << std::endl;
}
}
*/

cum_prof.resize(nhalos * (n_profbins+1));
for(unsigned long long int n=0;n<nhalos;n++){
	for(int j=0;j<n_profbins;j++){
		cum_prof[n*(n_profbins+1)+j+1] = cum_prof[n*(n_profbins+1)+j] + (unsigned int)prof[n*n_profbins+j];
	}
}

std::cerr << " Cumulative profile constructed.";


mass.resize(nhalos);
radius.resize(nhalos);

std::vector<double> logrbins(n_profbins+1);
std::vector<double> logdelta(n_profbins+1);
for(int j=0;j<n_profbins+1;j++){
	logrbins[j] = log(rbins[j]);
}
for(unsigned long long int n=0;n<nhalos;n++){
	// if(n%1000000==0) std::cerr << n << "\r" << std::flush;
	unsigned long long int istart = n*(n_profbins+1);
	while(cum_prof[istart]==0)istart++;
	if(istart >= (n+1)*(n_profbins+1)){
		std::cerr << "Error! Something is wrong at " << n << std::endl;
		mass[n] = radius[n] = 0.;
		continue;
	}
	int istart2 = istart%(n_profbins+1);
	if(cum_prof[istart]/cum_nmeans[istart2]<threshold){
		istart ++;
		istart2 ++;
		if(istart2 == n_profbins){
			mass[n] = radius[n] = 0.;
			continue;
		}
	}

	if(istart2 < n_profbins){
		int j(0);
		while(cum_prof[istart+j]/cum_nmeans[istart2+j]>threshold){
			j++;
			if(istart2+j>=n_profbins){
				mass[n] = radius[n] = 0.;
				break;
			}
		}
		double logdens_high = log(cum_prof[istart+j-1]/cum_nmeans[istart2+j-1]);
		double logdens_low = log(cum_prof[istart+j]/cum_nmeans[istart2+j]);
		double frac = (logdens_high-logthresh)/(logdens_high-logdens_low);
		double lograd = (1-frac)*log(rbins[istart2+j-1]) + frac * log(rbins[istart2+j]);
		radius[n] = exp(lograd);
		mass[n] = (float)(4.*M_PI/3.*radius[n]*radius[n]*radius[n]*mp*nbar*threshold);
		//correction
		mass[n] *= 1.+0.55*pow(mass[n]/mp,-0.6);
		radius[n] = pow(mass[n]/(4.*M_PI/3.*mp*nbar*threshold),1/3.);
	}
}

std::cerr << " Mass determined.";

std::vector<myhalo_full_str> hstr(nhalos);
for(unsigned long long int n=0;n<nhalos;n++){
	hstr[n].vmax = vmax[n];
	hstr[n].mass = mass[n];
	hstr[n].radius = radius[n];
	for(int j=0;j<3;j++){
		hstr[n].pos[j] = pos[3*n+j];
	}
	for(int j=0;j<n_profbins;j++){
		hstr[n].prof[j] = prof[(unsigned long long int)n_profbins*n+j];
	}
	int ix((floor)(pos[3*n  ]*ncell1d/Box));
	int iy((floor)(pos[3*n+1]*ncell1d/Box));
	int iz((floor)(pos[3*n+2]*ncell1d/Box));
	if(ix<0) ix += ncell1d;
	else if(ix>=ncell1d) ix -= ncell1d;
	if(iy<0) iy += ncell1d;
	else if(iy>=ncell1d) iy -= ncell1d;
	if(iz<0) iz += ncell1d;
	else if(iz>=ncell1d) iz -= ncell1d;
	hstr[n].cell = (ix * ncell1d + iy) * ncell1d + iz;
}

std::cerr << " Data copied to a structure.";

omptl::sort(hstr.begin(), hstr.end());

std::cerr << " Data sorted.";

std::vector<unsigned long long int>start_addr(ncell3d+1);

int cell_so_far(0);
for(unsigned long long int n=0;n<nhalos;n++){
	int cell_now(hstr[n].cell);
	if(cell_so_far!=cell_now){
		while(cell_so_far<cell_now){
			start_addr[++cell_so_far]=n;
		}
	}
}
start_addr[ncell3d] = nhalos;

std::cerr << " Cell address determined.";

for(unsigned long long int n=0;n<nhalos;n++){
	int cell_now(hstr[n].cell);
	int ix = cell_now/(ncell1d*ncell1d);
	int iy = (cell_now - ix*ncell1d*ncell1d)/ncell1d;
	int iz = cell_now - ix*ncell1d*ncell1d - iy*ncell1d;
	for(int iix=ix-1;iix<=ix+1;iix++){
		if(hstr[n].subhalo_flag) break;
		int iiix = (iix>=0) ? ((iix<ncell1d) ? iix : iix-ncell1d) : iix+ncell1d;
		for(int iiy=iy-1;iiy<=iy+1;iiy++){
			if(hstr[n].subhalo_flag) break;
			int iiiy = (iiy>=0) ? ((iiy<ncell1d) ? iiy : iiy-ncell1d) : iiy+ncell1d;
			for(int iiz=iz-1;iiz<=iz+1;iiz++){
				if(hstr[n].subhalo_flag) break;
				int iiiz = (iiz>=0) ? ((iiz<ncell1d) ? iiz : iiz-ncell1d) : iiz+ncell1d;
				int cell_targ((iiix*ncell1d+iiiy)*ncell1d+iiiz);
				if(cell_now == cell_targ){
					for(unsigned long long int 	m=n+1;m<start_addr[cell_targ+1];m++){
						float dx = hstr[n].pos[0] - hstr[m].pos[0];
						dx = (dx>-Box/2.) ? ((dx<Box/2.) ? dx : dx - Box) : dx + Box;
						float dx2 = dx*dx;
						if(dx2>hstr[m].radius*hstr[m].radius) continue;
						float dy = hstr[n].pos[1] - hstr[m].pos[1];
						dy = (dy>-Box/2.) ? ((dy<Box/2.) ? dy : dy - Box) : dy + Box;
						float dxy2 = dx2+dy*dy;
						if(dxy2>hstr[m].radius*hstr[m].radius) continue;
						float dz = hstr[n].pos[2] - hstr[m].pos[2];
						dz = (dz>-Box/2.) ? ((dz<Box/2.) ? dz : dz - Box) : dz + Box;
						float d2 = dxy2+dz*dz;
						if(d2 < hstr[m].radius*hstr[m].radius){
							hstr[n].subhalo_flag = true;
							break;
						}
					}
				}else{
					for(unsigned long long int 	m=start_addr[cell_targ];m<start_addr[cell_targ+1];m++){
						if(hstr[n].vmax>hstr[m].vmax) continue;
						float dx = hstr[n].pos[0] - hstr[m].pos[0];
						dx = (dx>-Box/2.) ? ((dx<Box/2.) ? dx : dx - Box) : dx + Box;
						float dx2 = dx*dx;
						if(dx2>hstr[m].radius*hstr[m].radius) continue;
						float dy = hstr[n].pos[1] - hstr[m].pos[1];
						dy = (dy>-Box/2.) ? ((dy<Box/2.) ? dy : dy - Box) : dy + Box;
						float dxy2 = dx2+dy*dy;
						if(dxy2>hstr[m].radius*hstr[m].radius) continue;
						float dz = hstr[n].pos[2] - hstr[m].pos[2];
						dz = (dz>-Box/2.) ? ((dz<Box/2.) ? dz : dz - Box) : dz + Box;
						float d2 = dxy2+dz*dz;
						if(d2 < hstr[m].radius*hstr[m].radius){
							hstr[n].subhalo_flag = true;
							break;
						}
					}
				}
			}
		}
	}
}

std::cerr << " Hosts/subs determined." << std::endl;


long long int nhosts(0);
for(long long int n=0;n<hstr.size();n++){
	if(!hstr[n].subhalo_flag) nhosts++;
}
std::cerr << nhosts << " host halos found." << std::endl;
hoststr.resize(nhosts);
nhosts=0;
//int nfake(0);
for(long long int n=0;n<hstr.size();n++){
	if(!hstr[n].subhalo_flag){
		if(hstr[n].mass < Mmin || hstr[n].mass >= Mmax) continue;
		hoststr[nhosts].mass = hstr[n].mass;
		for(int j=0;j<3;j++){
			hoststr[nhosts].pos[j] = hstr[n].pos[j];
		}
		for(int i=0;i<n_profbins;i++){
			hoststr[nhosts].prof[i] = hstr[n].prof[i];
		}
		nhosts++;
		//if(hstr[n].mass==0) nfake++;
	}
}

hoststr.resize(nhosts);
std::cerr << nhosts << " host halos fell in the mass range." << std::endl;
omptl::sort(hoststr.begin(), hoststr.end(), std::greater<>());

std::cerr << " Data sorted in descending order of mass." << std::endl;
//if(ntake2<nhosts){
//hoststr.resize(ntake2);
//std::cerr << ntake2 << " most massive halos selected." << std::endl;
//}
//std::cerr << "Least massive halo: " << hoststr[ntake2-1].mass << " M_sun/h" << std::endl;

xihm.resize(n_profbins);
r_direct.resize(n_profbins);
rmin_direct.resize(n_profbins);
rmax_direct.resize(n_profbins);
for(long long int n=0;n<hoststr.size();n++){
	for(int i=0;i<n_profbins;i++){
		xihm[i] += (double)hoststr[n].prof[i];
	}
}
for(int i=0;i<n_profbins;i++){
	rmin_direct[i] = rbins[i];
	rmax_direct[i] = rbins[i+1];
	r_direct[i] = rcens[i];
	xihm[i] = xihm[i] / (4.*M_PI/3. * (rbins[i+1] * rbins[i+1] * rbins[i+1] - rbins[i] * rbins[i] * rbins[i]) * nbar * (double)hoststr.size()) - 1.;
}

/*
for(int i=0;i<n_profbins;i++){
std::cout << rcens[i] << "  " << xihm[i] << std::endl;
}
*/
}

void stack_halo(double Box, int Npart1d, int ndom, std::vector<myhosthalo_full_str> &hoststr, double nh, std::vector<double>& rcens, std::vector<double>& rbins, std::vector<double> &xihm)
{

	int ntake((int)(nh*Box*Box*Box));
	double rmax(Box/(double)ndom); // The maximum radius up to which the profile is recorded.
	double rmin(rmax/1e4);

	double ratio = pow(rmax/rmin,1./n_profbins);
	rbins.resize(n_profbins+1);
	rcens.resize(n_profbins);
	for(int j=0;j<n_profbins+1;j++){
		rbins[j] = rmin * pow(ratio,j);
	}
	for(int j=0;j<n_profbins;j++){
		rcens[j] = rmin * pow(ratio,j+1/2.);
	}

	double nbar = (Npart1d/Box)*(Npart1d/Box)*(Npart1d/Box);

	xihm.resize(n_profbins);
	for(long long int n=0;n<ntake;n++){
		for(int i=0;i<n_profbins;i++){
			xihm[i] += (double)hoststr[n].prof[i];
		}
	}
	for(int i=0;i<n_profbins;i++){
		xihm[i] = xihm[i] / (4.*M_PI/3. * (rbins[i+1] * rbins[i+1] * rbins[i+1] - rbins[i] * rbins[i] * rbins[i]) * nbar * ntake) - 1.;
	}
	#if 0
	for(int i=0;i<n_profbins;i++){
	std::cout << rcens[i] << "  " << xihm[i] << std::endl;
}
#endif
}

void stack_halo(double Box, int Npart1d, int ndom, std::vector<myhosthalo_full_str> &hoststr, double nh, std::vector<double>& rcens, std::vector<double>& rbins, std::vector<double> &xihm, int nsub, std::vector<double> &xihm_JK)
{
	int nsub3d(nsub*nsub*nsub);
	int ntake((int)(nh*Box*Box*Box));
	double rmax(Box/(double)ndom); // The maximum radius up to which the profile is recorded.
	double rmin(rmax/1e4);

	double ratio = pow(rmax/rmin,1./n_profbins);
	rbins.resize(n_profbins+1);
	rcens.resize(n_profbins);
	for(int j=0;j<n_profbins+1;j++){
		rbins[j] = rmin * pow(ratio,j);
	}
	for(int j=0;j<n_profbins;j++){
		rcens[j] = rmin * pow(ratio,j+1/2.);
	}

	double nbar = (Npart1d/Box)*(Npart1d/Box)*(Npart1d/Box);

	xihm.resize(n_profbins);
	xihm_JK.resize(nsub3d*n_profbins);

	std::vector<int> nhalos_sub(nsub3d,0);



	for(long long int n=0;n<ntake;n++){
		for(int i=0;i<n_profbins;i++){
			xihm[i] += (double)hoststr[n].prof[i];
		}
		int ix_sub = hoststr[n].pos[0] * nsub / Box;
		if(ix_sub<0) ix_sub += nsub;
		else if(ix_sub>=nsub) ix_sub -= nsub;
		int iy_sub = hoststr[n].pos[1] * nsub / Box;
		if(iy_sub<0) iy_sub += nsub;
		else if(iy_sub>=nsub) iy_sub -= nsub;
		int iz_sub = hoststr[n].pos[2] * nsub / Box;
		if(iz_sub<0) iz_sub += nsub;
		else if(iz_sub>=nsub) iz_sub -= nsub;
		int reg_sub((ix_sub*nsub+iy_sub)*nsub+iz_sub);
		for(int j=0;j<nsub3d;j++){
			if(reg_sub != j){
				nhalos_sub[j] ++;
				for(int i=0;i<n_profbins;i++){
					xihm_JK[j*n_profbins+i] += (double)hoststr[n].prof[i];
				}
			}
		}
	}
	for(int i=0;i<n_profbins;i++){
		xihm[i] = xihm[i] / (4.*M_PI/3. * (rbins[i+1] * rbins[i+1] * rbins[i+1] - rbins[i] * rbins[i] * rbins[i]) * nbar * ntake) - 1.;
		for(int j=0;j<nsub3d;j++){
			xihm_JK[j*n_profbins+i] = xihm_JK[j*n_profbins+i] / (4.*M_PI/3. * (rbins[i+1] * rbins[i+1] * rbins[i+1] - rbins[i] * rbins[i] * rbins[i]) * nbar * nhalos_sub[j]) - 1.;
		}
	}
}
