#ifndef CLIQSPSPARSE_HPP
#define CLIQSPSPARSE_HPP

#include "subProblem.hpp"
#include "myUtils.hpp"
#include <string>

int performSPFwdSparse(const subProblem *, const std::vector<int> &, const std::vector<double> &, const std::vector<double> &, const double, const std::vector<double> &, double &, std::vector<double> &);

int performSPBwdSparse(const subProblem *, const std::vector<int> &, const std::vector<double> &, const double, std::vector<double> &, std::vector<double> &);

int cliqSPSparse(const subProblem *subProb, const double tau, double &energy, std::vector<double> &nodeMarg)
{
 std::vector<int> subProbNodeOffset = subProb->getNodeOffset();
 std::vector<double> subProbDualVar = subProb->getDualVar();

 std::vector<double> expoMaxBwd;
 std::vector<double> bwdMargVec;

 performSPBwdSparse(subProb, subProbNodeOffset, subProbDualVar, tau, bwdMargVec, expoMaxBwd);

 performSPFwdSparse(subProb, subProbNodeOffset, subProbDualVar, bwdMargVec, tau, expoMaxBwd, energy, nodeMarg);

 return 0;
}

//nodes, messages etc concerning messages coming from/going to left/top is referred with prefix of one
//and coming from/going to right/bottom with prefix of two

int performSPBwdSparse(const subProblem *subProb, const std::vector<int> &subProbNodeOffset, const std::vector<double> &subProbDualVar, const double tau, std::vector<double> &bwdMargVec, std::vector<double> &expoMaxBwd)
{
 int firstSetInd = 0;
 int secSetInd = 1;

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
 std::vector<double> expoVecSum;
 std::vector<double> expoVecSparse;
 std::vector<double> expoVecOffset;

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

  expoVecSum.push_back(expo);

  if (expoMaxInitFlag) {
   expoMax = expo;
   expoMaxInitFlag = false;
  }
  else if (expo > expoMax) {
   expoMax = expo;
  }

 } //for iTwoLab = [0:nTwoSetLab)

 expoMaxBwd.resize(nOneSetLab, expoMax);

 for (std::set<int>::const_iterator iCliqLab = sparseLab.begin(); iCliqLab != sparseLab.end(); ++iCliqLab) {//performSPBwdSparse
  double dualSum = 0;

  for (std::vector<int>::iterator iSet = twoSet.begin(); iSet != twoSet.end(); ++iSet) {   
   varAssign = static_cast<int>(floor(*iCliqLab/stride[*iSet])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];
  }

  expo = tau*(subProb->getCE(*iCliqLab) - dualSum); //$$$$

  expoVecSparse.push_back(expo);

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

  expoVecOffset.push_back(expo);
 } //for iCliqLab

 margConstSum = 0;

 for (int iTwoLab = 0; iTwoLab != nTwoSetLab; ++iTwoLab) {

  expVal = exp(expoVecSum[iTwoLab] - expoMax);
  myUtils::checkRangeError(expVal);

  margConstSum += expVal;
 } //for iTwoLab = [0:nTwoSetLab)

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

 return 0;
}

int performSPFwdSparse(const subProblem *subProb, const std::vector<int> &subProbNodeOffset, const std::vector<double> &subProbDualVar, const std::vector<double> &bwdMargVec, const double tau, const std::vector<double> &expoMaxBwd, double &energy, std::vector<double> &nodeMarg)
{
 energy = 0;

 int firstSetInd = 0;
 int secSetInd = 1;

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

 double expoMax = 0, expoMaxOne = 0, expoMaxTwo = 0;
 std::vector<double> expoMaxFwd;

 std::vector<double> expoVecSum;

 std::vector<double> expoVecOne;
 std::vector<double> expoVecTwo;
 std::vector<double> expoVecSparse;
 std::vector<double> expoVecOffset;

 int varAssign;

 bool expoMaxInitFlag = true;

 bool expoMaxOneInitFlag = true;
 bool expoMaxTwoInitFlag = true;

 for (int iOneLab = 0; iOneLab != nOneSetLab; ++iOneLab) {//performSPFwdSparse
  int oneStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(iOneLab/oneStride[oneStrideInd])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++oneStrideInd;
  }
 
  expo = tau*(cEnergyConst - dualSum);

  expoVecSum.push_back(expo);

  if (expoMaxInitFlag) {
   expoMax = expo;
   expoMaxInitFlag = false;
  }
  else if (expo > expoMax) {
   expoMax = expo;
  }

  expo = -1*tau*dualSum + log(bwdMargVec[iOneLab]) + expoMaxBwd[iOneLab];
  //expo = -1*tau*dualSum;

  expoVecOne.push_back(expo);

  if (expoMaxOneInitFlag) {
   expoMaxOne = expo;
   expoMaxOneInitFlag = false;
  }
  else if (expo > expoMaxOne) {
   expoMaxOne = expo;
  }

 } //for iOneLab = [0:nOneSetLab)

 expoMaxFwd.resize(nTwoSetLab, expoMax);

 for (std::set<int>::const_iterator iCliqLab = sparseLab.begin(); iCliqLab != sparseLab.end(); ++iCliqLab) {//performSPFwdSparse
  int oneStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(*iCliqLab/stride[*iSet])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++oneStrideInd;
  }

  expo = tau*(subProb->getCE(*iCliqLab) - dualSum); //$$$$

  expoVecSparse.push_back(expo);

  int fwdMargInd = 0;
  int twoStrideInd = 0;

  for (std::vector<int>::iterator iSet = twoSet.begin(); iSet != twoSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(*iCliqLab/stride[*iSet])) % label[*iSet];

   fwdMargInd += varAssign*twoStride[twoStrideInd];

   ++twoStrideInd;
  }

  if (expo > expoMaxFwd[fwdMargInd]) { //$$$$
   expoMaxFwd[fwdMargInd] = expo;
  }

  expo = tau*(cEnergyConst - dualSum);

  expoVecOffset.push_back(expo);
 } //for iCliqLab

 double margConstSum = 0;

 double energyExpSumOne = 0;

 for (int iOneLab = 0; iOneLab != nOneSetLab; ++iOneLab) {
  expVal = exp(expoVecSum[iOneLab] - expoMax);
  myUtils::checkRangeError(expVal);

  margConstSum += expVal;

  expVal = exp(expoVecOne[iOneLab] - expoMaxOne);
  myUtils::checkRangeError(expVal);

  //expVal *= bwdMargVec[iOneLab];

  int oneStrideInd = 0;

  for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(iOneLab/oneStride[oneStrideInd])) % label[*iSet];

   nodeMarg[subProbNodeOffset[*iSet] + varAssign] += expVal; 

   ++oneStrideInd;
  }

  energyExpSumOne += expVal;
 } //for iOneLab = [0:nOneSetLab)

 std::vector<double> fwdMargVec(nTwoSetLab, margConstSum);

 for (int iTwoLab = 0; iTwoLab != nTwoSetLab; ++iTwoLab) {
  fwdMargVec[iTwoLab] *= exp(expoMax - expoMaxFwd[iTwoLab]);
 }

 int iterCnt = 0;
 
 for (std::set<int>::const_iterator iCliqLab = sparseLab.begin(); iCliqLab != sparseLab.end(); ++iCliqLab) {//performSPFwdSparse
  int fwdMargInd = 0;
  int twoStrideInd = 0;

  for (std::vector<int>::iterator iSet = twoSet.begin(); iSet != twoSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(*iCliqLab/stride[*iSet])) % label[*iSet];

   fwdMargInd += varAssign*twoStride[twoStrideInd];

   ++twoStrideInd;
  }

  expVal = exp(expoVecSparse[iterCnt] - expoMaxFwd[fwdMargInd]);
  myUtils::checkRangeError(expVal);

  double expValOff = exp(expoVecOffset[iterCnt] - expoMaxFwd[fwdMargInd]);
  myUtils::checkRangeError(expVal);

  fwdMargVec[fwdMargInd] += expVal - expValOff;

  ++iterCnt;
 } //for iCliqLab

 for (int iTwoLab = 0; iTwoLab != nTwoSetLab; ++iTwoLab) {//performSPFwdSparse
  int twoStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = twoSet.begin(); iSet != twoSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(iTwoLab/twoStride[twoStrideInd])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++twoStrideInd;
  }

  expo = -1*tau*dualSum + log(fwdMargVec[iTwoLab]) + expoMaxFwd[iTwoLab];
  //expo = -1*tau*dualSum;

  expoVecTwo.push_back(expo);

  if (expoMaxTwoInitFlag) {
   expoMaxTwo = expo;
   expoMaxTwoInitFlag = false;
  }
  else if (expo > expoMaxTwo) {
   expoMaxTwo = expo;
  }

 } //for iTwoLab = [0:nTwoSetLab)

 double energyExpSumTwo = 0;

 for (int iTwoLab = 0; iTwoLab != nTwoSetLab; ++iTwoLab) {//performSPFwdSparse
  expVal = exp(expoVecTwo[iTwoLab] - expoMaxTwo);
  myUtils::checkRangeError(expVal);

  int twoStrideInd = 0;

  for (std::vector<int>::iterator iSet = twoSet.begin(); iSet != twoSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(iTwoLab/twoStride[twoStrideInd])) % label[*iSet];

   //nodeMarg[subProbNodeOffset[*iSet] + varAssign] += expVal*exp(expoMaxTwo - expoMaxOne);
   nodeMarg[subProbNodeOffset[*iSet] + varAssign] += expVal;

   ++twoStrideInd;
  }

  energyExpSumTwo += expVal;
 } //for iTwoLab = [0:nTwoSetLab)

// for (std::vector<double>::iterator nodeLabIter = nodeMarg.begin(); nodeLabIter != nodeMarg.end(); ++nodeLabIter) {
//  *nodeLabIter /= energyExpSumOne;
// }

 for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
  for (int iL = 0; iL != label[*iSet]; ++iL) {
   nodeMarg[subProbNodeOffset[*iSet] + iL] /= energyExpSumOne;
  }
 }

 for (std::vector<int>::iterator iSet = twoSet.begin(); iSet != twoSet.end(); ++iSet) {
  for (int iL = 0; iL != label[*iSet]; ++iL) {
   nodeMarg[subProbNodeOffset[*iSet] + iL] /= energyExpSumTwo;
  }
 }

 energy = (1/tau)*(log(energyExpSumOne) + expoMaxOne);

 double energyDebug = (1/tau)*(log(energyExpSumTwo) + expoMaxTwo);

 return 0;
}

#endif // CLIQSPSPARSE_HPP
