#ifndef CLIQSPSPARSESCD_HPP
#define CLIQSPSPARSESCD_HPP

#include "subProblem.hpp"
#include "myUtils.hpp"
#include <string>

int performSPFwdSparseSCD(const subProblem *, const std::vector<int> &, const std::vector<double> &, const double, const std::vector<double> &, const std::vector<double> &, const int &, const int &, std::vector<double> &);

int performSPBwdSparseSCD(const subProblem *, const std::vector<int> &, const std::vector<double> &, const double, const int &, const int &, std::vector<double> &, std::vector<double> &);

int cliqSPSparseSCD(const subProblem *subProb, const double tau, const int &nodeInd, std::vector<double> &nodeMarg)
{
 std::vector<int> subProbNodeOffset = subProb->getNodeOffset();
 std::vector<double> subProbDualVar = subProb->getDualVar();

 std::vector<double> expoMaxBwd; //double *expoMaxBwd;
 std::vector<double> bwdMargVec; //double *bwdMargVec;

 double debugTime = myUtils::getTime();

 int firstSetInd = 0;
 int secSetInd = 1;

 std::vector<int> oneSet = subProb->getSet(firstSetInd);
 std::vector<int> twoSet = subProb->getSet(secSetInd); 

 std::vector<int>::iterator oneNodeIter = std::find(oneSet.begin(), oneSet.end(), nodeInd);
 std::vector<int>::iterator twoNodeIter = std::find(twoSet.begin(), twoSet.end(), nodeInd);

 if ((oneNodeIter == oneSet.end()) && (twoNodeIter == twoSet.end())) {
  std::cout<<"cliqSPSparseSCD::ERROR: incorrect node index."<<std::endl; 
 }
 else if (twoNodeIter != twoSet.end()) {
  firstSetInd = 1;
  secSetInd = 0;  
 }

 performSPBwdSparseSCD(subProb, subProbNodeOffset, subProbDualVar, tau, firstSetInd, secSetInd, bwdMargVec, expoMaxBwd);

 //std::cout<<"Backward pass took "<<myUtils::getTime()-debugTime<<std::endl;

 debugTime = myUtils::getTime();

 performSPFwdSparseSCD(subProb, subProbNodeOffset, subProbDualVar, tau, bwdMargVec, expoMaxBwd, firstSetInd, secSetInd, nodeMarg);

 //std::cout<<"Forward pass took "<<myUtils::getTime()-debugTime<<std::endl;

 //delete [] expoMaxBwd;
 //delete [] bwdMargVec;

 return 0;
}

//nodes, messages etc concerning messages coming from/going to left/top is referred with prefix of one
//and coming from/going to right/bottom with prefix of two

int performSPBwdSparseSCD(const subProblem *subProb, const std::vector<int> &subProbNodeOffset, const std::vector<double> &subProbDualVar, const double tau, const int &firstSetInd, const int &secSetInd, std::vector<double> &bwdMargVec, std::vector<double> &expoMaxBwd)
{
 //int firstSetInd = 0;
 //int secSetInd = 1;

 std::vector<int> oneSet = subProb->getSet(firstSetInd);
 std::vector<int> twoSet = subProb->getSet(secSetInd); //message is from set "two" to set "one"
 std::vector<short> label = subProb->getNodeLabel();
 std::vector<int> stride = subProb->getStride();
 std::vector<int> oneStride = subProb->getStride(firstSetInd);
 std::vector<int> twoStride = subProb->getStride(secSetInd);

 int nOneSetLab = subProb->getSetLabCnt(firstSetInd);
 int nTwoSetLab = subProb->getSetLabCnt(secSetInd);
 std::set<int> sparseLab = subProb->getSparseInd();
 double cEnergyConst = subProb->getCEConst();

 //std::vector<double> expoVecSum;
 //std::vector<double> expoVecSparse;
 //std::vector<double> expoVecOffset;

 double* expoVecSum = new double[nTwoSetLab];
 double* expoVecSparse;
 double* expoVecOffset;

 double expoMax;
 bool expoMaxInitFlag = true;
 double margConstSum;

 double expo = 0, expVal = 0;
 int varAssign;

 for (int iTwoLab = 0; iTwoLab != nTwoSetLab; ++iTwoLab) {//performSPBwdSparse
  int twoStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = twoSet.begin(); iSet != twoSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(iTwoLab/twoStride[twoStrideInd])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++twoStrideInd;
  }
 
  expo = tau*(cEnergyConst - dualSum);

  expoVecSum[iTwoLab] = expo;

  if (expoMaxInitFlag) {
   expoMax = expo;
   expoMaxInitFlag = false;
  }
  else if (expo > expoMax) {
   expoMax = expo;
  }

 } //for iTwoLab = [0:nTwoSetLab)

 //expoMaxBwd = new double[nOneSetLab];
 //std::fill_n(expoMaxBwd, nOneSetLab, expoMax);

 expoMaxBwd.resize(nOneSetLab,expoMax);

 expoVecSparse = new double[sparseLab.size()];
 expoVecOffset = new double[sparseLab.size()];

 int sparseIndCnt = 0;

 for (std::set<int>::const_iterator iCliqLab = sparseLab.begin(); iCliqLab != sparseLab.end(); ++iCliqLab) {//performSPBwdSparse
  int twoStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = twoSet.begin(); iSet != twoSet.end(); ++iSet) {   
   varAssign = static_cast<int>(floor(*iCliqLab/stride[*iSet])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++twoStrideInd;
  }

  expo = tau*(subProb->getCE(*iCliqLab) - dualSum); //$$$$

  expoVecSparse[sparseIndCnt] = expo;

  int bwdMargInd = 0;
  int oneStrideInd = 0;

  for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(*iCliqLab/stride[*iSet])) % label[*iSet];

   bwdMargInd += varAssign*oneStride[oneStrideInd];

   ++oneStrideInd;
  }

  if (expo > expoMaxBwd[bwdMargInd]) { //$$$$
   expoMaxBwd[bwdMargInd] = expo;
  }

  expo = tau*(cEnergyConst - dualSum);

  expoVecOffset[sparseIndCnt] = expo;

  ++sparseIndCnt;
 } //for iCliqLab

 margConstSum = 0;

 for (int iTwoLab = 0; iTwoLab != nTwoSetLab; ++iTwoLab) {

  expVal = exp(expoVecSum[iTwoLab] - expoMax);
  myUtils::checkRangeError(expVal);

  margConstSum += expVal;
 } //for iTwoLab = [0:nTwoSetLab)

 //bwdMargVec = new double[nOneSetLab];
 bwdMargVec.resize(nOneSetLab);

 for (int iOneLab = 0; iOneLab != nOneSetLab; ++iOneLab) {
  bwdMargVec[iOneLab] = margConstSum*exp(expoMax - expoMaxBwd[iOneLab]);
 }

 //bwdMargVec.resize(nOneSetLab, margConstSum);

 int iterCnt = 0;

 for (std::set<int>::const_iterator iCliqLab = sparseLab.begin(); iCliqLab != sparseLab.end(); ++iCliqLab) {//performSPBwdSparse
  int bwdMargInd = 0;
  int oneStrideInd = 0;

  for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(*iCliqLab/stride[*iSet])) % label[*iSet];

   bwdMargInd += varAssign*oneStride[oneStrideInd];

   ++oneStrideInd;
  }

  expVal = exp(expoVecSparse[iterCnt] - expoMaxBwd[bwdMargInd]);
  myUtils::checkRangeError(expVal);

  double expValOff = exp(expoVecOffset[iterCnt] - expoMaxBwd[bwdMargInd]);
  myUtils::checkRangeError(expVal);

  bwdMargVec[bwdMargInd] += expVal - expValOff;

  ++iterCnt;
 } //for iCliqLab

 delete [] expoVecSum;
 delete [] expoVecSparse;
 delete [] expoVecOffset;

 return 0;
}

int performSPFwdSparseSCD(const subProblem *subProb, const std::vector<int> &subProbNodeOffset, const std::vector<double> &subProbDualVar, const double tau, const std::vector<double> &bwdMargVec, const std::vector<double> &expoMaxBwd, const int &firstSetInd, const int &secSetInd, std::vector<double> &nodeMarg)
{
 //energy = 0;

 //int firstSetInd = 0;
 //int secSetInd = 1;

 int nOneSetLab = subProb->getSetLabCnt(firstSetInd);
 int nTwoSetLab = subProb->getSetLabCnt(secSetInd);
 std::vector<int> oneSet = subProb->getSet(firstSetInd);
 std::vector<int> twoSet = subProb->getSet(secSetInd);
 std::vector<short> label = subProb->getNodeLabel();

 std::vector<int> stride = subProb->getStride();
 std::vector<int> oneStride = subProb->getStride(firstSetInd);
 std::vector<int> twoStride = subProb->getStride(secSetInd);
 double cEnergyConst = subProb->getCEConst();
 std::set<int> sparseLab = subProb->getSparseInd();

 double expo = 0, expVal = 0;

 double expoMaxOne = 0;

 double* expoVecOne;

 int varAssign;

 bool expoMaxInitFlag = true;

 bool expoMaxOneInitFlag = true;

 expoVecOne = new double[nOneSetLab];

 for (int iOneLab = 0; iOneLab != nOneSetLab; ++iOneLab) {//performSPFwdSparse
  int oneStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(iOneLab/oneStride[oneStrideInd])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++oneStrideInd;
  }

  expo = -1*tau*dualSum + log(bwdMargVec[iOneLab]) + expoMaxBwd[iOneLab];

  expoVecOne[iOneLab] = expo;

  if (expoMaxOneInitFlag) {
   expoMaxOne = expo;
   expoMaxOneInitFlag = false;
  }
  else if (expo > expoMaxOne) {
   expoMaxOne = expo;
  }

 } //for iOneLab = [0:nOneSetLab)

 double energyExpSumOne = 0;

 for (int iOneLab = 0; iOneLab != nOneSetLab; ++iOneLab) {
  expVal = exp(expoVecOne[iOneLab] - expoMaxOne);
  myUtils::checkRangeError(expVal);

  int oneStrideInd = 0;

  for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(iOneLab/oneStride[oneStrideInd])) % label[*iSet];

   nodeMarg[subProbNodeOffset[*iSet] + varAssign] += expVal; 

   ++oneStrideInd;
  }

  energyExpSumOne += expVal;
 } //for iOneLab = [0:nOneSetLab)

 for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
  for (int iL = 0; iL != label[*iSet]; ++iL) {
   nodeMarg[subProbNodeOffset[*iSet] + iL] /= energyExpSumOne;
  }
 }

 delete [] expoVecOne;

 return 0;
}

#endif // CLIQSPSPARSESCD_HPP
