#ifndef CLIQSPSPARSEGRADENERGY_HPP
#define CLIQSPSPARSEGRADENERGY_HPP

#include "subProblem.hpp"
#include "myUtils.hpp"
#include <string>

int performSPFwdSparse(const subProblem *, const std::vector<int> &, const std::vector<double> &, const double, const std::vector<double> &, const std::vector<double> &, double &, std::vector<double> &);

int performSPBwdSparse(const subProblem *, const std::vector<int> &, const std::vector<double> &, const double, std::vector<double> &, std::vector<double> &);

int cliqSPSparseGradEnergy(const subProblem *subProb, const std::vector<double> &subProbDualVar, const double tau, double &energy, std::vector<double> &nodeMarg)
{
 std::vector<int> subProbNodeOffset = subProb->getNodeOffset();

 std::vector<double> expoMaxBwd;
 std::vector<double> bwdMargVec;

 double debugTime = myUtils::getTime();

 performSPBwdSparse(subProb, subProbNodeOffset, subProbDualVar, tau, bwdMargVec, expoMaxBwd);

 //std::cout<<"Backward pass took "<<myUtils::getTime()-debugTime<<std::endl;

 debugTime = myUtils::getTime();

 performSPFwdSparse(subProb, subProbNodeOffset, subProbDualVar, tau, bwdMargVec, expoMaxBwd, energy, nodeMarg);

 //std::cout<<"Forward pass took "<<myUtils::getTime()-debugTime<<std::endl;

 //delete [] expoMaxBwd;
 //delete [] bwdMargVec;

 return 0;
}

#endif //CLIQSPSPARSEGRADENERGY_HPP
