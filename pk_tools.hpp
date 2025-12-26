#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <Eigen/Dense>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

void ps_fileload(std::string fname, Eigen::VectorXd &pk);
void mwc_fileload(std::string Mfname, std::string Wfname, std::string Cfname, Eigen::MatrixXd &M, Eigen::MatrixXd &W, Eigen::MatrixXd &C);
void chi_square(Eigen::VectorXd pk, Eigen::MatrixXd M, Eigen::MatrixXd W, Eigen::MatrixXd C, BinnedData pk0, BinnedData pk2, BinnedData pk4, double &x2, double kmax);
void mcmc(double chi2, double& delta_v, double& v_th, std::vector<double>& chi2list, std::vector<double>& dvlist, std::vector<double>& vthlist, gsl_rng* rand_ins, int k);

void ps_fileload(std::string fname, Eigen::VectorXd &pk){
    Eigen::VectorXd k_ps(400);
    Eigen::VectorXd k_eff(400);
    Eigen::VectorXd pk0(400);
    Eigen::VectorXd pk1(400);
    Eigen::VectorXd pk2(400);
    Eigen::VectorXd pk3(400);
    Eigen::VectorXd pk4(400);
    Eigen::VectorXd sig0(400);
    Eigen::VectorXd sig1(400);
    Eigen::VectorXd sig2(400);
    Eigen::VectorXd sig3(400);
    Eigen::VectorXd sig4(400);
    Eigen::VectorXd modes(400);

    int within_header = 0;
    std::ifstream file(fname);
    std::cout << "check pkfilename :" << fname << std::endl;
    std::cout << "load BOSS pk file ... ";
    std::string line;
    int i = 0;

    while(std::getline(file, line)) {
        if (within_header < 2){
            if(line == "### header ###"){
                within_header += 1;
            }
        } else{
            std::istringstream iss(line);
            iss >> k_ps(i);
            iss >> k_eff(i);
            iss >> pk0(i);
            iss >> sig0(i);
            iss >> pk1(i);
            iss >> sig1(i);
            iss >> pk2(i);
            iss >> sig2(i);
            iss >> pk3(i);
            iss >> sig3(i);
            iss >> pk4(i);
            iss >> sig4(i);
            iss >> modes(i);
            i++;
        }

    }
    file.close();
    std::cout << "done." << std::endl;

    using RowMajorMatrix40x10 = Eigen::Matrix<double, 40, 10, Eigen::RowMajor>;

    Eigen::Map<RowMajorMatrix40x10>weight(modes.data());
    Eigen::Map<RowMajorMatrix40x10>pk0mat(pk0.data());
    Eigen::Map<RowMajorMatrix40x10>pk1mat(pk1.data());
    Eigen::Map<RowMajorMatrix40x10>pk2mat(pk2.data());
    Eigen::Map<RowMajorMatrix40x10>pk3mat(pk3.data());
    Eigen::Map<RowMajorMatrix40x10>pk4mat(pk4.data());

    Eigen::VectorXd wsum = weight.rowwise().sum();
    Eigen::VectorXd wmean0 = pk0mat.cwiseProduct(weight).rowwise().sum().cwiseQuotient(wsum);
    Eigen::VectorXd wmean2 = pk2mat.cwiseProduct(weight).rowwise().sum().cwiseQuotient(wsum);
    Eigen::VectorXd wmean4 = pk4mat.cwiseProduct(weight).rowwise().sum().cwiseQuotient(wsum);

    pk << wmean0, wmean2, wmean4;

}

void mwc_fileload(std::string Mfname, std::string Wfname, std::string Cfname, Eigen::MatrixXd &M, Eigen::MatrixXd &W, Eigen::MatrixXd &C){
    M.setZero();
    W.setZero();
    C.setZero();

    std::cout <<  "load M file ... ";
    std::ifstream Mfile(Mfname);
    std::string line;
    for (int i = 0; i < 2000; i++){
        std::getline(Mfile, line);
        std::istringstream issm(line);
        for(int j = 0; j < 1200; j++){
            issm >> M(i, j);
        }
    }
    Mfile.close();
    std::cout <<  "done." << std::endl;
    std::cout <<  "M rows: " << M.rows() << std::endl;
    std::cout <<  "M cols: " << M.cols() << std::endl;
    //std::cout << "M(469, 1271)" << M(469, 1271) << std::endl;

    std::cout <<  "load W file ... ";
    std::ifstream Wfile(Wfname);
    for(int i = 0; i < 200; i++){
        std::getline(Wfile, line);
        std::istringstream issw(line);
        for(int j = 0; j < 2000; j++){
            issw >> W(i, j);
        }
    }
    Wfile.close();
    std::cout <<  "done." << std::endl;
    std::cout <<  "W rows: " << W.rows() << std::endl;
    std::cout <<  "W cols: " << W.cols() << std::endl;
    // std::cout << "W :" << W << std::endl;

    std::cout <<  "load C file ... ";
    std::ifstream Cfile(Cfname);

    Eigen::VectorXd dummy(25600);

    int ii = 0;
    int jj = 0;
    int k = 0;
    for(int i = 0; i < 200; i++){
        std::getline(Cfile, line);
        std::istringstream issc(line);
        jj = 0;
        for(int j = 0; j < 200; j++){
            if((i >= 40 && i < 80) || (i >= 120 && i < 160) || (j >=40 && j< 80) || (j >= 120 && j < 160)){
                issc >> dummy(k);
                k++;
            } else {
                issc >> C(ii, jj);
                //std::cout << "ii, jj: " << ii << ", " << jj << std::endl;
                jj++;
            }
        }
        if(i < 40 || (i >= 80 && i < 120) || i >= 160){
            ii++;
        }
    }
    Cfile.close();
    std::cout <<  "done." << std::endl;
    std::cout <<  "C rows: " << C.rows() << std::endl;
    std::cout <<  "C cols: " << C.cols() << std::endl;
    // std::cout << "C" << C << std::endl;
}

void chi_square(Eigen::VectorXd Bpk, Eigen::MatrixXd M, Eigen::MatrixXd W, Eigen::MatrixXd C, BinnedData pk0, BinnedData pk2, BinnedData pk4, double &x2, double kmax){
    Eigen::VectorXd pk(1200);
    Eigen::VectorXd wmp(200);
    Eigen::MatrixXd Cinv;
    Eigen::VectorXd psim0(pk0.get_nbin());
    Eigen::VectorXd psim2(pk2.get_nbin());
    Eigen::VectorXd psim4(pk4.get_nbin());
    Eigen::VectorXd psim(120);

    for (int i = 0; i < pk0.get_nbin(); i++){
        psim0(i) = pk0.get_ymean(i);
        psim2(i) = pk2.get_ymean(i);
        psim4(i) = pk4.get_ymean(i);
    }

    //std::cout << "pksim0 size: " << psim0.size() << std::endl;
    //std::cout << "pksim2 size: " << psim2.size() << std::endl;
    //std::cout << "pksim4 size: " << psim4.size() << std::endl;

    Cinv = C.inverse();
    //std::cout <<  "Cinv rows: " << Cinv.rows() << std::endl;
    //std::cout << "Cinv cols: " << Cinv.cols() << std::endl;
    pk << psim0, psim2, psim4;
    //std::cout << "pk size: " << pk.size() << std::endl;
    wmp = W*M*pk;
    int j = 0;
    for (int i = 0; i < 200; i++){
        if(i < 40 || (i >= 80 && i < 120) || i >= 160){
            psim(j) = wmp(i);
            j++;
        }
    }
    //std::cout <<  "wmp size: " << wmp.size() << std::endl;
    //std::cout << "psim size: " << psim.size() << std::endl;
    // std::cout << "Bpk :" << Bpk << std::endl;

    for (int i=0; i < Bpk.size(); i++){
        for (int j=0; j < Bpk.size(); j++){
            x2 += (Bpk(i)-psim(i))*Cinv(i,j)*(Bpk(j)-psim(j));
        }
    }
    // std::cout << "Cinv" << Cinv << std::endl;
}

void mcmc(double chi2, double& delta_v, double& v_th, std::vector<double>& chi2list, std::vector<double>& dvlist, std::vector<double>& vthlist, gsl_rng* rand_ins, int k){
    if((chi2 < chi2list[k-1]) || k == 0) {
        chi2list[k] = chi2;
        dvlist[k] = delta_v;
        vthlist[k] = v_th;
        do {
            delta_v = delta_v + gsl_ran_gaussian(rand_ins, 4.0);
            v_th = v_th + gsl_ran_gaussian(rand_ins, 4.0);
        } while(delta_v<0);
        std::cout << "if number: " << 0 << std::endl;
    } else {
        double r = std::exp(-(chi2-chi2list[k-1])/2);
        double rand_num = gsl_rng_uniform(rand_ins);
        if(r > rand_num) {
            chi2list[k] = chi2;
            dvlist[k] = delta_v;
            vthlist[k] = v_th;
            do {
                delta_v = delta_v + gsl_ran_gaussian(rand_ins, 4.0);
                v_th = v_th + gsl_ran_gaussian(rand_ins, 4.0);
            } while(delta_v<0);
            std::cout << "if number: " << 1 << std::endl;
        } else {
            chi2list[k] = chi2list[k-1];
            dvlist[k] = dvlist[k-1];
            vthlist[k] = vthlist[k-1];
            std::cout << "dvlist[k-1] : " << dvlist[k-1] << std::endl;
            std::cout << "vthlist[k-1] : " << vthlist[k-1] << std::endl;
            //std::cout << "gsl_ran_gaussian" << gsl_ran_gaussian(rand_ins, 4.0) << std::endl;
            double last_dv = dvlist[k-1];
            double last_vth = vthlist[k-1];
            double new_dv;
            double new_vth;
            do {
                new_dv = last_dv + gsl_ran_gaussian(rand_ins, 4.0);
                new_vth = last_vth + gsl_ran_gaussian(rand_ins, 4.0);
                std::cout << "new_dv : " << new_dv <<std::endl;
                std::cout << "new_vth : " << new_vth <<std::endl;
            } while(new_dv < 0);
            delta_v = new_dv;
            v_th = new_vth;
            std::cout << "mcmc delta_v : " << delta_v << std::endl;
            std::cout << "mcmc v_th : " << v_th << std::endl;
            std::cout << "if number: " << 2 << std::endl;
        }
    }
}
