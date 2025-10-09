#pragma once
#include <cassert>
#include "util.hpp"

void get_file_format(std::string fname, int& nlines, int& ncolumns);

void load_part_ascii(std::string FileName, std::vector<float>& px, std::vector<float>& py, std::vector<float>& pz, std::vector<float>& mass, int xcol, int ycol, int zcol, int mcol, float mth) {
    std::string tmp, token;
    std::istringstream stream;
    std::ifstream fin;
    int nlines, ncolumns;
    std::string::size_type comment_start = 0;
    get_file_format(FileName, nlines, ncolumns);

    assert(ncolumns >= std::max(std::max(std::max(xcol, ycol), zcol), mcol));

    px.resize(nlines);
    py.resize(nlines);
    pz.resize(nlines);
    mass.resize(nlines);
    nlines = 0;

    fin.open(FileName.c_str());
    while (!fin.eof()) {
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
			if(nelem==xcol+1) px[nlines] = atof(token.c_str());
			if(nelem==ycol+1) py[nlines] = atof(token.c_str());
			if(nelem==zcol+1) pz[nlines] = atof(token.c_str());
			if(nelem==mcol+1) mass[nlines] = atof(token.c_str());
		}

		if (mass[nlines] > mth) {
			nlines++;
		} else {
			px.pop_back();
			py.pop_back();
			pz.pop_back();
			mass.pop_back();
			}
		}
	}
	fin.close();
	std::cerr << "### Done. " << nlines << " haloes read ###" << std::endl;
}

void load_part_ascii(std::string FileName, std::vector<float>& px, std::vector<float>& py, std::vector<float>& pz, std::vector<float>& mass, std::vector<float>&ax_ratio, std::vector<float>&theta, int xcol, int ycol, int zcol, int mcol, int axcol, int angcol, float mth) {
    std::string tmp, token;
    std::istringstream stream;
    std::ifstream fin;
    int nlines, ncolumns;
    std::string::size_type comment_start = 0;
    get_file_format(FileName, nlines, ncolumns);

    assert(ncolumns >= std::max(std::max(std::max(std::max(std::max(xcol, ycol), zcol), mcol), axcol), angcol));

    px.resize(nlines);
    py.resize(nlines);
    pz.resize(nlines);
    mass.resize(nlines);
    ax_ratio.resize(nlines);
    theta.resize(nlines);
    nlines = 0;

    fin.open(FileName.c_str());
    while (!fin.eof()) {
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

			if(nelem==xcol+1) px[nlines] = atof(token.c_str());
			if(nelem==ycol+1) py[nlines] = atof(token.c_str());
			if(nelem==zcol+1) pz[nlines] = atof(token.c_str());
			if(nelem==mcol+1) mass[nlines] = atof(token.c_str());
			if(nelem==axcol+1) ax_ratio[nlines] = atof(token.c_str());
			if(nelem==angcol+1) theta[nlines] = atof(token.c_str());
		}

		if (mass[nlines] > mth) {
			nlines++;
		} else {
			px.pop_back();
			py.pop_back();
			pz.pop_back();
			mass.pop_back();
			ax_ratio.pop_back();
			theta.pop_back();
			}
		}
	}
	fin.close();
	std::cerr << "### Done. " << nlines << " haloes read ###" << std::endl;
}

void load_part_ascii(std::string FileName, std::vector<float>&px, std::vector<float>&py, std::vector<float>&pz, std::vector<float>&mass){

	std::string tmp, token;
	std::istringstream stream;
	std::ifstream fin;
	int nlines, ncolumns;
	std::string::size_type comment_start = 0;
	get_file_format(FileName, nlines, ncolumns);

	assert(ncolumns == 4);
	px.resize(nlines);
	py.resize(nlines);
	pz.resize(nlines);
	mass.resize(nlines);
	nlines = 0;

	fin.open(FileName.c_str());
	while(!fin.eof()){
		std::getline(fin,tmp);
		if( (comment_start = tmp.find("#")) != std::string::size_type(-1) )
			tmp = tmp.substr(0, comment_start);
		std::istringstream stream(tmp);
		int nelem = 0;
		while (stream >> token) nelem++;
		if(nelem==ncolumns){
			std::istringstream stream_read(tmp);
			nelem=0;
			while(stream_read >> token){
				nelem++;
				if(nelem==1) px[nlines] = atof(token.c_str());
				if(nelem==2) py[nlines] = atof(token.c_str());
				if(nelem==3) pz[nlines] = atof(token.c_str());
				if(nelem==4) mass[nlines] = atof(token.c_str());
			}
			nlines++;
		}
	}
	fin.close();
	std::cerr << " done. " << nlines << " halos read." << std::endl;
}

void load_part_ascii(std::string FileName, std::vector<float>&px, std::vector<float>&py, std::vector<float>&pz, std::vector<float>&mass, std::vector<float>&theta){

	std::string tmp, token;
	std::istringstream stream;
	std::ifstream fin;
	int nlines, ncolumns;
	std::string::size_type comment_start = 0;
	get_file_format(FileName,nlines,ncolumns);

	assert(ncolumns == 5);
	px.resize(nlines);
	py.resize(nlines);
	pz.resize(nlines);
	mass.resize(nlines);
	theta.resize(nlines);
	nlines = 0;

	fin.open(FileName.c_str());
	while(!fin.eof()){
		std::getline(fin,tmp);
		if( (comment_start = tmp.find("#")) != std::string::size_type(-1) )
			tmp = tmp.substr(0, comment_start);
		std::istringstream stream(tmp);
		int nelem = 0;
		while (stream >> token) nelem++;
		if(nelem==ncolumns){
			std::istringstream stream_read(tmp);
			nelem=0;
			while(stream_read >> token){
				nelem++;
				if(nelem==1) px[nlines] = atof(token.c_str());
				if(nelem==2) py[nlines] = atof(token.c_str());
				if(nelem==3) pz[nlines] = atof(token.c_str());
				if(nelem==4) mass[nlines] = atof(token.c_str());
				if(nelem==5) theta[nlines] = atof(token.c_str());
			}
			nlines++;
		}
	}
	fin.close();
	std::cerr << " done. " << nlines << " halos read." << std::endl;
}

void load_part_ascii(std::string FileName, std::vector<float>&px, std::vector<float>&py, std::vector<float>&pz, std::vector<float>&mass, std::vector<float>&ax_ratio, std::vector<float>&theta){

	std::string tmp, token;
	std::istringstream stream;
	std::ifstream fin;
	int nlines, ncolumns;
	std::string::size_type comment_start = 0;
	get_file_format(FileName,nlines,ncolumns);

	assert(ncolumns == 6);
	px.resize(nlines);
	py.resize(nlines);
	pz.resize(nlines);
	mass.resize(nlines);
	ax_ratio.resize(nlines);
	theta.resize(nlines);
	nlines = 0;

	fin.open(FileName.c_str());
	while(!fin.eof()){
		std::getline(fin,tmp);
		if( (comment_start = tmp.find("#")) != std::string::size_type(-1) )
			tmp = tmp.substr(0, comment_start);
		std::istringstream stream(tmp);
		int nelem = 0;
		while (stream >> token) nelem++;
		if(nelem==ncolumns){
			std::istringstream stream_read(tmp);
			nelem=0;
			while(stream_read >> token){
				nelem++;
				if(nelem==1) px[nlines] = atof(token.c_str());
				if(nelem==2) py[nlines] = atof(token.c_str());
				if(nelem==3) pz[nlines] = atof(token.c_str());
				if(nelem==4) mass[nlines] = atof(token.c_str());
				if(nelem==5) ax_ratio[nlines] = atof(token.c_str());
				if(nelem==6) theta[nlines] = atof(token.c_str());
			}
			nlines++;
		}
	}
	fin.close();
	std::cerr << " done. " << nlines << " halos read." << std::endl;
}



inline void get_file_format(std::string fname, int& nlines, int& ncolumns){
	ncolumns = nlines = 0;
	std::string::size_type comment_start = 0;
	std::string tmp;
	std::string token;
	std::ifstream fin;
	fin.open(fname.c_str());
	while(!fin.eof()){
		std::getline(fin,tmp);
		if( (comment_start = tmp.find("#")) != std::string::size_type(-1) )
			tmp = tmp.substr(0, comment_start);
		std::istringstream stream(tmp);
		int nelem = 0;
		while (stream >> token) nelem++;
		if(nelem>0) nlines++;
		if(ncolumns !=0 && ncolumns<nelem){
			std::cerr << "Invalid file format. Aborting." << std::endl;
			exit(1);
		}else if(ncolumns<nelem){
			ncolumns = nelem;
		}
	}
	fin.close();
}

void load_halo(std::string FileBase, std::vector<float>&px, std::vector<float>&py, std::vector<float>&pz){
	std::ifstream fin;
	std::string buf;
	int nfiles(0);
	size_t nhalo(0);

	while(1){
		buf = FileBase + "_vmax."+ itos(nfiles);
		if (!fexists(buf)) break;
		nfiles++;
		nhalo += get_fs(buf)/sizeof(float);
	}
	std::cout << nhalo << " halos stored in " << nfiles << " files." << std::endl;

	px.resize(nhalo);
	py.resize(nhalo);
	pz.resize(nhalo);

	size_t fsize(0);
	size_t nhalo_tmp(0);
	size_t nhalo_cum(0);

	std::vector<float> farray;
	farray.resize(4*nhalo*3/nfiles);

	for(auto i=0;i<nfiles;i++){
		buf = FileBase + "_pos."+ itos(i);
		fsize = get_fs(buf);
		nhalo_tmp = fsize/sizeof(float)/3;
		fin.open(buf.c_str());
		fin.read((char *)&farray[0],fsize);
		fin.close();
		for(auto j=0;j<nhalo_tmp;j++){
			px[nhalo_cum+j] = farray[3*j  ];
			py[nhalo_cum+j] = farray[3*j+1];
			pz[nhalo_cum+j] = farray[3*j+2];
		}
		nhalo_cum += nhalo_tmp;
	}
	assert(nhalo_cum == nhalo);
	std::cerr << "Loading halo positions done. " << nhalo << " halos read." << std::endl;
}


void load_halo(std::string FileBase, std::vector<float>&px, std::vector<float>&py, std::vector<float>&pz, std::vector<float>&vx, std::vector<float>&vy, std::vector<float>&vz){
	std::ifstream fin;
	std::string buf;
	int nfiles(0);
	size_t nhalo(0);

	while(1){
		buf = FileBase + "_vmax."+ itos(nfiles);
		if (!fexists(buf)) break;
		nfiles++;
		nhalo += get_fs(buf)/sizeof(float);
	}
	std::cout << nhalo << " halos stored in " << nfiles << " files." << std::endl;

	px.resize(nhalo);
	py.resize(nhalo);
	pz.resize(nhalo);
	vx.resize(nhalo);
	vy.resize(nhalo);
	vz.resize(nhalo);

	size_t fsize(0);
	size_t nhalo_tmp(0);
	size_t nhalo_cum(0);

	std::vector<float> farray;
	farray.resize(4*nhalo*3/nfiles);

	for(auto i=0;i<nfiles;i++){
		buf = FileBase + "_pos."+ itos(i);
		fsize = get_fs(buf);
		nhalo_tmp = fsize/sizeof(float)/3;
		fin.open(buf.c_str());
		fin.read((char *)&farray[0],fsize);
		fin.close();
		for(auto j=0;j<nhalo_tmp;j++){
			px[nhalo_cum+j] = farray[3*j  ];
			py[nhalo_cum+j] = farray[3*j+1];
			pz[nhalo_cum+j] = farray[3*j+2];
		}
		nhalo_cum += nhalo_tmp;
	}
	assert(nhalo_cum == nhalo);
	nhalo_cum = 0;
	for(auto i=0;i<nfiles;i++){
		buf = FileBase + "_bulkvel."+ itos(i);
		fsize = get_fs(buf);
		nhalo_tmp = fsize/sizeof(float)/3;
		fin.open(buf.c_str());
		fin.read((char *)&farray[0],fsize);
		fin.close();
		for(auto j=0;j<nhalo_tmp;j++){
			vx[nhalo_cum+j] = farray[3*j  ];
			vy[nhalo_cum+j] = farray[3*j+1];
			vz[nhalo_cum+j] = farray[3*j+2];
		}
		nhalo_cum += nhalo_tmp;
	}
	assert(nhalo_cum == nhalo);
	std::cerr << "Loading halo positions and velocities done. " << nhalo << " halos read." << std::endl;
}

void load_halo(std::string FileBase, std::vector<float>&px, std::vector<float>&py, std::vector<float>&pz, std::vector<float>&vx, std::vector<float>&vy, std::vector<float>&vz, std::vector<float>&vmax){
	std::ifstream fin;
	std::string buf;
	int nfiles(0);
	size_t nhalo(0);

	while(1){
		buf = FileBase + "_vmax."+ itos(nfiles);
		if (!fexists(buf)) break;
		nfiles++;
		nhalo += get_fs(buf)/sizeof(float);
	}
	std::cout << nhalo << " halos stored in " << nfiles << " files." << std::endl;

	px.resize(nhalo);
	py.resize(nhalo);
	pz.resize(nhalo);
	vx.resize(nhalo);
	vy.resize(nhalo);
	vz.resize(nhalo);
	vmax.resize(nhalo);

	size_t fsize(0);
	size_t nhalo_tmp(0);
	size_t nhalo_cum(0);

	std::vector<float> farray;
	farray.resize(4*nhalo*3/nfiles);

	for(auto i=0;i<nfiles;i++){
		buf = FileBase + "_pos."+ itos(i);
		fsize = get_fs(buf);
		nhalo_tmp = fsize/sizeof(float)/3;
		fin.open(buf.c_str());
		fin.read((char *)&farray[0],fsize);
		fin.close();
		for(auto j=0;j<nhalo_tmp;j++){
			px[nhalo_cum+j] = farray[3*j  ];
			py[nhalo_cum+j] = farray[3*j+1];
			pz[nhalo_cum+j] = farray[3*j+2];
		}
		nhalo_cum += nhalo_tmp;
	}
	assert(nhalo_cum == nhalo);
	nhalo_cum = 0;
	for(auto i=0;i<nfiles;i++){
		buf = FileBase + "_bulkvel."+ itos(i);
		fsize = get_fs(buf);
		nhalo_tmp = fsize/sizeof(float)/3;
		fin.open(buf.c_str());
		fin.read((char *)&farray[0],fsize);
		fin.close();
		for(auto j=0;j<nhalo_tmp;j++){
			vx[nhalo_cum+j] = farray[3*j  ];
			vy[nhalo_cum+j] = farray[3*j+1];
			vz[nhalo_cum+j] = farray[3*j+2];
		}
		nhalo_cum += nhalo_tmp;
	}
	assert(nhalo_cum == nhalo);
	nhalo_cum = 0;
	for(auto i=0;i<nfiles;i++){
		buf = FileBase + "_vmax."+ itos(i);
		fsize = get_fs(buf);
		nhalo_tmp = fsize/sizeof(float);
		fin.open(buf.c_str());
		fin.read((char *)&farray[0],fsize);
		fin.close();
		for(auto j=0;j<nhalo_tmp;j++){
			vmax[nhalo_cum+j] = farray[j];
		}
		nhalo_cum += nhalo_tmp;
	}
	assert(nhalo_cum == nhalo);

	std::cerr << "Loading halo positions, velocities and maximum circular velocities done. " << nhalo << " halos read." << std::endl;
	
}

