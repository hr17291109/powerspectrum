#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <fftw3.h>
#include "binneddata.hpp"
#include "fielddata.hpp"
#include "fileloader.hpp"
#include "cosmo.hpp"
#include "util.hpp"
#include "binning.hpp"
#include "param.hpp"
#include "filenames.hpp"

int main(int argc, char **argv){
    std::string FileBase = argv[1];
    int snapnum(atoi(argv[2]));
    std::string OutBase = argv[3];

    // Cosmological parameters
    double Omega_cb; // This is Omega_c + Omega_b, not Omega_m
    double Omega_m;  // Omega_m (=Omega_cb + Omega_{massive-neutrino})
    double h0;       // Dimensionless hubble parameter (should be around 0.7)
    double As;       // Amplitude of the primordial fluctuations
    double ns;       // Spectral index
    double k0;       // The "pivot" wavenumber at which As is given

    // Simulation parameters
    double Box;      // in Mpc/h
    double redshift; // The output redshift
    int ng(2000); // Number of grid points per dim for FFT
    int Npart1d;

    switch(snapnum){
        case 0:
            redshift = 0.61;
            break;
        case 1:
            redshift = 0.51;
            break;
        case 2:
            redshift = 0.38;
            break;
        default:
            redshift = 0;
            break;
    }
    std::cerr << "Redshift: " << redshift << std::endl;

    param::parameter p1(FileBase+"/"+params_dir+"/"+nugenic_param_file);
    Omega_cb = p1.get<double>("Omega");
    Box = p1.get<double>("Box");
    Npart1d = (long long int)p1.get<int>("Npart");
    As = p1.get<double>("As1");
    ns = p1.get<double>("ns1");
    h0 = p1.get<double>("HubbleParam");
    k0 = p1.get<double>("kpivot")/h0;

    std::cout << "Info from " << FileBase+"/"+params_dir+"/"+nugenic_param_file << std::endl;
    std::cout << "Box: " << Box << std::endl;
    std::cerr << "Npart1d: " << Npart1d << std::endl;
    std::cout << "Omega_cb: " << Omega_cb << std::endl;
    std::cout << "As: " << As << std::endl;
    std::cout << "ns: " << ns << std::endl;
    std::cout << "h: " << h0 << std::endl;
    std::cout << "kpivot: " << k0 << std::endl;

    param::parameter p2(FileBase+"/"+params_dir+"/"+class_param_file);

    Omega_m = p2.get<double>("Omega_m");

    std::cout << "Info from " << FileBase+"/"+params_dir+"/"+class_param_file << std::endl;
    std::cout << "Omega_m: " << Omega_m << std::endl << std::endl;

    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    std::cout << omp_get_max_threads() << " threads will be used in FFT." << std::endl;
    // fftwf_import_wisdom_from_filename(wisdom_file.c_str());

    // FieldData Df_lin(ng,Box,true);
    // fftwf_export_wisdom_to_filename(wisdom_file.c_str());
    // Df_lin.load(FileBase+"/"+white_dir+"/"+white_file);

    std::cerr << "### Load halos from " << FileBase+"/halo_catalog" << " ###" << std::endl;
    std::vector<myhosthalo_str> halos;
    std::vector<myhosthalo_full_str> halos_full;

    load_halo_full_vmaxthreshold(FileBase, snapnum, halos_full);

    halos.resize(halos_full.size());
    for(long long int i=0;i<halos.size();i++){
            halos[i].mass = halos_full[i].mass;
            for(int j=0;j<3;j++){
                    halos[i].pos[j] = halos_full[i].pos[j];
                    halos[i].vel[j] = halos_full[i].vel[j];
            }
    }

    double sfac(get_sfac(redshift,FileBase+"/"+trans_dir+"/"+expansion_file));

    FieldData halo_overdensity(ng,Box,false);
    halo_overdensity.assignment(halos,false,false,sfac,2);
    halo_overdensity.do_fft();

    // Monopole moment
    int ell = 0;
    BinnedData pk0 = halo_overdensity.calc_power(nbins, kmin, kmax, logbin, ell);
    pk0.dump(OutBase+"_pk0.dat");

    // Quadrupole moment
    ell = 2;
    BinnedData pk2 = halo_overdensity.calc_power(nbins, kmin, kmax, logbin, ell);
    pk2.dump(OutBase+"_pk2.dat");

    // Hexadecapole moment
    ell = 4;
    BinnedData pk4 = halo_overdensity.calc_power(nbins, kmin, kmax, logbin, ell);
    pk4.dump(OutBase+"_pk4.dat");

    exit(0);
}
