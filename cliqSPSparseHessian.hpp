#ifndef CLIQSPSPARSEHESSIAN_HPP
#define CLIQSPSPARSEHESSIAN_HPP

#include "subProblem.hpp"
#include "myUtils.hpp"
#include <string>
#include <thread>

int performSPBwdSparseHessian(const subProblem *, const std::vector<int> &, const std::vector<double> &, const int &, const int &, const double&, std::vector<double> &, std::vector<double> &);

int performSPFwdNodeMarg(const subProblem *, const std::vector<int> &, const std::vector<double> &, const int &, const int &, const double &, const std::vector<double> &, const std::vector<double> &, double &, std::vector<double> &, std::vector<double> &);

int performSPFwdNodePairMargOnly(const subProblem *, const std::vector<int> &, const std::vector<double> &, const int &, const int &, const double &, const std::vector<double> &, const std::vector<double> &, std::vector<double> &);

int cliqSPSparseHessian(const subProblem *subProb, const double tau, double &energy, std::vector<double> &nodeMarg, std::vector<double> &nodePairMarg)
{
 std::vector<int> subProbNodeOffset = subProb->getNodeOffset();
 std::vector<double> subProbDualVar = subProb->getDualVar();

 //std::vector<double> expoMaxBwdOne;
 //std::vector<double> bwdMargVecOne;
 //std::vector<double> expoMaxBwdTwo;
 //std::vector<double> bwdMargVecTwo;
 //std::vector<double> expoMaxBwdThree;
 //std::vector<double> bwdMargVecThree;

 std::vector<std::vector<double> > expoMaxBwd(3);
 std::vector<std::vector<double> > bwdMargVec(3);

 //double debugTime = myUtils::getTime();

 int firstSetInd = 0;
 int secSetInd = 1;
 std::thread threadOne = std::thread(performSPBwdSparseHessian,subProb, subProbNodeOffset, subProbDualVar, firstSetInd, secSetInd, tau, std::ref(bwdMargVec[0]), std::ref(expoMaxBwd[0]));
 firstSetInd = 2;
 secSetInd = 3;
 std::thread threadTwo = std::thread(performSPBwdSparseHessian,subProb, subProbNodeOffset, subProbDualVar, firstSetInd, secSetInd, tau, std::ref(bwdMargVec[1]), std::ref(expoMaxBwd[1]));
 firstSetInd = 4;
 secSetInd = 5;
 std::thread threadThree = std::thread(performSPBwdSparseHessian,subProb, subProbNodeOffset, subProbDualVar, firstSetInd, secSetInd, tau, std::ref(bwdMargVec[2]), std::ref(expoMaxBwd[2]));

 threadOne.join();
 threadTwo.join();
 threadThree.join();

 //std::cout<<"Backward pass took "<<myUtils::getTime() - debugTime<<std::endl;
 //debugTime = myUtils::getTime();

 firstSetInd = 0;
 secSetInd = 1;
 threadOne = std::thread(performSPFwdNodeMarg, subProb, subProbNodeOffset, subProbDualVar, firstSetInd, secSetInd, tau, bwdMargVec[0], expoMaxBwd[0], std::ref(energy), std::ref(nodeMarg), std::ref(nodePairMarg));
 firstSetInd = 2;
 secSetInd = 3;
 threadTwo = std::thread(performSPFwdNodePairMargOnly, subProb, subProbNodeOffset, subProbDualVar, firstSetInd, secSetInd, tau, bwdMargVec[1], expoMaxBwd[1], std::ref(nodePairMarg));
 firstSetInd = 4;
 secSetInd = 5;
 threadThree = std::thread(performSPFwdNodePairMargOnly,subProb, subProbNodeOffset, subProbDualVar, firstSetInd, secSetInd, tau, bwdMargVec[2], expoMaxBwd[2], std::ref(nodePairMarg));

 threadOne.join();
 threadTwo.join();
 threadThree.join();

 //std::cout<<"Forward pass took "<<myUtils::getTime() - debugTime<<std::endl;

 return 0;
}

int performSPBwdSparseHessian(const subProblem *subProb, const std::vector<int> &subProbNodeOffset, const std::vector<double> &subProbDualVar, const int &firstSetInd, const int &secSetInd, const double &tau, std::vector<double> &bwdMargVec, std::vector<double> &expoMaxBwd)
{
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

int performSPFwdNodeMarg(const subProblem *subProb, const std::vector<int> &subProbNodeOffset, const std::vector<double> &subProbDualVar, const int &firstSetInd, const int &secSetInd, const double &tau, const std::vector<double> &bwdMargVec, const std::vector<double> &expoMaxBwd, double &energy, std::vector<double> &nodeMarg, std::vector<double> &nodePairMarg)
{
 energy = 0;

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

 int subDualSiz = subProb->getDualSiz();

 double expo = 0, expVal = 0;

 double expoMax = 0, expoMaxOne = 0, expoMaxTwo = 0;
 std::vector<double> expoMaxFwd;

 //std::vector<double> expoVecSum;
 //std::vector<double> expoVecOne;
 //std::vector<double> expoVecTwo;
 //std::vector<double> expoVecSparse;
 //std::vector<double> expoVecOffset;

 double* expoVecSum;
 double* expoVecOne;
 double* expoVecTwo;
 double* expoVecSparse;
 double* expoVecOffset;

 int varAssign;

 bool expoMaxInitFlag = true;

 bool expoMaxOneInitFlag = true;
 bool expoMaxTwoInitFlag = true;

 expoVecSum = new double[nOneSetLab];
 expoVecOne = new double[nOneSetLab];

 for (int iOneLab = 0; iOneLab != nOneSetLab; ++iOneLab) {//performSPFwdSparse
  int oneStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(iOneLab/oneStride[oneStrideInd])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++oneStrideInd;
  }

  expo = tau*(cEnergyConst - dualSum);

  expoVecSum[iOneLab] = expo;

  if (expoMaxInitFlag) {
   expoMax = expo;
   expoMaxInitFlag = false;
  }
  else if (expo > expoMax) {
   expoMax = expo;
  }

  double logVal = log(bwdMargVec[iOneLab]);
  myUtils::checkLogError(logVal);

  expo = -1*tau*dualSum + logVal + expoMaxBwd[iOneLab];
  //expo = -1*tau*dualSum;

  expoVecOne[iOneLab] = expo;

  if (expoMaxOneInitFlag) {
   expoMaxOne = expo;
   expoMaxOneInitFlag = false;
  }
  else if (expo > expoMaxOne) {
   expoMaxOne = expo;
  }

 } //for iOneLab = [0:nOneSetLab)

 expoMaxFwd.resize(nTwoSetLab, expoMax);

 expoVecSparse = new double[sparseLab.size()];
 expoVecOffset = new double[sparseLab.size()];

 int sparseIndCnt = 0;

 for (std::set<int>::const_iterator iCliqLab = sparseLab.begin(); iCliqLab != sparseLab.end(); ++iCliqLab) {//performSPFwdSparse
  int oneStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(*iCliqLab/stride[*iSet])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++oneStrideInd;
  }

  expo = tau*(subProb->getCE(*iCliqLab) - dualSum); //$$$$

  expoVecSparse[sparseIndCnt] = expo;

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

  expoVecOffset[sparseIndCnt] = expo;

  ++sparseIndCnt;
 } //for iCliqLab

 std::set<int> onePairMargInd;
 std::set<int> twoPairMargInd;

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

   int nodePos = *iSet;

   int pairStrideInd = 0;

   for (std::vector<int>::iterator jSet = oneSet.begin(); jSet != iSet; ++jSet) {
    int pairVarAssign = static_cast<int>(iOneLab/oneStride[pairStrideInd]) % label[*jSet];

    int pairNodePos = *jSet;

    nodePairMarg[(subProbNodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + subProbNodeOffset[nodePos] + varAssign] += expVal;
    //nodePairMargTest += expVal;

    //std::cout<<"CLIQSPNODEMARG: INDEX: "<<(subProbNodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + subProbNodeOffset[nodePos] + varAssign<<" NODES "<<pairNodePos<<" "<<nodePos<<" LABEL "<<pairVarAssign<<" "<<varAssign<<" VALUE "<<expVal<<std::endl;

    onePairMargInd.insert((subProbNodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + subProbNodeOffset[nodePos] + varAssign);

    ++pairStrideInd;
   }

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

 expoVecTwo = new double[nTwoSetLab];

 for (int iTwoLab = 0; iTwoLab != nTwoSetLab; ++iTwoLab) {//performSPFwdSparse
  int twoStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = twoSet.begin(); iSet != twoSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(iTwoLab/twoStride[twoStrideInd])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++twoStrideInd;
  }

  double logVal = log(fwdMargVec[iTwoLab]);
  myUtils::checkLogError(logVal);

  expo = -1*tau*dualSum + logVal + expoMaxFwd[iTwoLab];
  //expo = -1*tau*dualSum;

  expoVecTwo[iTwoLab] = expo;

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

   int nodePos = *iSet;

   int pairStrideInd = 0;

   for (std::vector<int>::iterator jSet = twoSet.begin(); jSet != iSet; ++jSet) {
    int pairVarAssign = static_cast<int>(iTwoLab/twoStride[pairStrideInd]) % label[*jSet];

    int pairNodePos = *jSet;

    nodePairMarg[(subProbNodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + subProbNodeOffset[nodePos] + varAssign] += expVal;
    //nodePairMargTest += expVal;

    //std::cout<<"CLIQSPNODEMARG: INDEX: "<<(subProbNodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + subProbNodeOffset[nodePos] + varAssign<<" NODES "<<pairNodePos<<" "<<nodePos<<" LABEL "<<pairVarAssign<<" "<<varAssign<<" VALUE "<<expVal<<std::endl;

    twoPairMargInd.insert((subProbNodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + subProbNodeOffset[nodePos] + varAssign);

    ++pairStrideInd;
   }

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

 for (std::set<int>::iterator iSet = onePairMargInd.begin(); iSet != onePairMargInd.end(); ++iSet) {
  nodePairMarg[*iSet] /= energyExpSumOne;
 }

 for (std::set<int>::iterator iSet = twoPairMargInd.begin(); iSet != twoPairMargInd.end(); ++iSet) {
  nodePairMarg[*iSet] /= energyExpSumTwo;
 }

 energy = (1/tau)*(log(energyExpSumOne) + expoMaxOne);

 //double energyDebug = (1/tau)*(log(energyExpSumTwo) + expoMaxTwo);

 //std::cout<<"CLIQSPNODEMARG: ENERGY ONE "<<energy<<" TWO "<<energyDebug<<std::endl;

 delete [] expoVecSum;
 delete [] expoVecOne;
 delete [] expoVecTwo;
 delete [] expoVecSparse;
 delete [] expoVecOffset;

 return 0;
}

int performSPFwdNodePairMargOnly(const subProblem *subProb, const std::vector<int> &subProbNodeOffset, const std::vector<double> &subProbDualVar, const int &firstSetInd, const int &secSetInd, const double &tau, const std::vector<double> &bwdMargVec, const std::vector<double> &expoMaxBwd, std::vector<double> &nodePairMarg)
{
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

 std::vector<int> nodeOffset = subProb->getNodeOffset();

 int subDualSiz = subProb->getDualSiz();

 int blkSiz = subProb->getBlkSiz();

 double expo = 0, expVal = 0;

 double expoMax = 0, expoMaxOne = 0, expoMaxTwo = 0;
 std::vector<double> expoMaxFwd;

 //std::vector<double> expoVecSum;
 //std::vector<double> expoVecOne;
 //std::vector<double> expoVecTwo;
 //std::vector<double> expoVecSparse;
 //std::vector<double> expoVecOffset;

 double* expoVecSum;
 double* expoVecOne;
 double* expoVecTwo;
 double* expoVecSparse;
 double* expoVecOffset;

 int varAssign;

 bool expoMaxInitFlag = true;

 bool expoMaxOneInitFlag = true;
 bool expoMaxTwoInitFlag = true;

 expoVecSum = new double[nOneSetLab];
 expoVecOne = new double[nOneSetLab];

 for (int iOneLab = 0; iOneLab != nOneSetLab; ++iOneLab) {//performSPFwdSparse
  int oneStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(iOneLab/oneStride[oneStrideInd])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++oneStrideInd;
  }

  expo = tau*(cEnergyConst - dualSum);

  expoVecSum[iOneLab] = expo;

  if (expoMaxInitFlag) {
   expoMax = expo;
   expoMaxInitFlag = false;
  }
  else if (expo > expoMax) {
   expoMax = expo;
  }

  double logVal = log(bwdMargVec[iOneLab]);
  myUtils::checkLogError(logVal);

  expo = -1*tau*dualSum + logVal + expoMaxBwd[iOneLab];
  //expo = -1*tau*dualSum;

  expoVecOne[iOneLab] = expo;

  if (expoMaxOneInitFlag) {
   expoMaxOne = expo;
   expoMaxOneInitFlag = false;
  }
  else if (expo > expoMaxOne) {
   expoMaxOne = expo;
  }

 } //for iOneLab = [0:nOneSetLab)

 expoMaxFwd.resize(nTwoSetLab, expoMax);

 expoVecSparse = new double[sparseLab.size()];
 expoVecOffset = new double[sparseLab.size()];

 int sparseIndCnt = 0;

 for (std::set<int>::const_iterator iCliqLab = sparseLab.begin(); iCliqLab != sparseLab.end(); ++iCliqLab) {//performSPFwdSparse
  int oneStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = oneSet.begin(); iSet != oneSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(*iCliqLab/stride[*iSet])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++oneStrideInd;
  }

  expo = tau*(subProb->getCE(*iCliqLab) - dualSum); //$$$$

  expoVecSparse[sparseIndCnt] = expo;

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

  expoVecOffset[sparseIndCnt] = expo;

  ++sparseIndCnt;
 } //for iCliqLab

 double margConstSum = 0;

 double energyExpSumOne = 0;

 std::set<int> onePairMargInd;
 std::set<int> twoPairMargInd;

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

   int nodePos = *iSet;

   int pairStrideInd = 0;

   for (std::vector<int>::iterator jSet = oneSet.begin(); jSet != iSet; ++jSet) {
    int pairVarAssign = static_cast<int>(floor(iOneLab/oneStride[pairStrideInd])) % label[*jSet];

    int pairNodePos = *jSet;

    nodePairMarg[(nodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + nodeOffset[nodePos] + varAssign] += expVal;
    //nodePairMargTest += expVal;

    //std::cout<<"CLIQSPNODEPAIR: INDEX: "<<(nodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + nodeOffset[nodePos] + varAssign<<" NODES "<<pairNodePos<<" "<<nodePos<<" LABELS "<<pairVarAssign<<" "<<varAssign<<" VALUE "<<expVal<<std::endl;

    onePairMargInd.insert((nodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + nodeOffset[nodePos] + varAssign);

    ++pairStrideInd;
   }

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

 expoVecTwo = new double[nTwoSetLab];

 for (int iTwoLab = 0; iTwoLab != nTwoSetLab; ++iTwoLab) {//performSPFwdSparse
  int twoStrideInd = 0;

  double dualSum = 0;

  for (std::vector<int>::iterator iSet = twoSet.begin(); iSet != twoSet.end(); ++iSet) {
   varAssign = static_cast<int>(floor(iTwoLab/twoStride[twoStrideInd])) % label[*iSet];

   dualSum += subProbDualVar[subProbNodeOffset[*iSet] + varAssign];

   ++twoStrideInd;
  }

  double logVal = log(fwdMargVec[iTwoLab]);
  myUtils::checkLogError(logVal);

  expo = -1*tau*dualSum + logVal + expoMaxFwd[iTwoLab];
  //expo = -1*tau*dualSum;

  expoVecTwo[iTwoLab] = expo;

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

   int nodePos = *iSet;

   int pairStrideInd = 0;

   for (std::vector<int>::iterator jSet = twoSet.begin(); jSet != iSet; ++jSet) {
    int pairVarAssign = static_cast<int>(floor(iTwoLab/twoStride[pairStrideInd])) % label[*jSet];

    int pairNodePos = *jSet;

    nodePairMarg[(nodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + nodeOffset[nodePos] + varAssign] += expVal;
    //nodePairMargTest += expVal;

    //std::cout<<"CLIQSPNODEPAIR: INDEX: "<<(nodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + nodeOffset[nodePos] + varAssign<<" NODES "<<pairNodePos<<" "<<nodePos<<" LABELS "<<pairVarAssign<<" "<<varAssign<<" VALUE "<<expVal<<std::endl;

    twoPairMargInd.insert((nodeOffset[pairNodePos] + pairVarAssign)*subDualSiz + nodeOffset[nodePos] + varAssign);

    ++pairStrideInd;
   }

   ++twoStrideInd;
  }

  energyExpSumTwo += expVal;
 } //for iTwoLab = [0:nTwoSetLab)

// for (std::vector<double>::iterator nodeLabIter = nodeMarg.begin(); nodeLabIter != nodeMarg.end(); ++nodeLabIter) {
//  *nodeLabIter /= energyExpSumOne;
// }

 for (std::set<int>::iterator iSet = onePairMargInd.begin(); iSet != onePairMargInd.end(); ++iSet) {
  nodePairMarg[*iSet] /= energyExpSumOne;
 }

 for (std::set<int>::iterator iSet = twoPairMargInd.begin(); iSet != twoPairMargInd.end(); ++iSet) {
  nodePairMarg[*iSet] /= energyExpSumTwo;
 }

 //std::cout<<"CLIQSPNODEPAIR: ENERGY ONE "<<(1/tau)*(log(energyExpSumOne) + expoMaxOne)<<" TWO "<<(1/tau)*(log(energyExpSumTwo) + expoMaxTwo)<<std::endl;

 delete [] expoVecSum;
 delete [] expoVecOne;
 delete [] expoVecTwo;
 delete [] expoVecSparse;
 delete [] expoVecOffset;

 return 0;
}

#endif // CLIQSPSPARSE_HPP
