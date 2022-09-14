#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>

#include "solve_scd.h"

#include "cliq_sparse_scd.h"
#include "../dual_sys.h"
#include "../myUtils.hpp"

int SolveSCD::solve(DualSys* dual_sys)
{
 std::cout<<"SolveSCD: Sparse flag "<<dual_sys->sparseFlag_<<std::endl;

 double tCompEnergies = 0;

 bool contIter = true;

 int cntIter = 0, cntIterTau = 0;

 double gradBasedDamp = 0;

 double bestPrimalEnergy = 0;

 dual_sys->timeStart_ = myUtils::getTime();

 while ((contIter) && (cntIter < dual_sys->maxIter_)) {
  double tFull = myUtils::getTime();

  ++cntIter;
  ++cntIterTau;

  //updateMSD(cntIter);
  updateStar(dual_sys, cntIter);

  dual_sys->popGradEnergy();

  double gradNorm = myUtils::norm<double>(dual_sys->gradient_,
                                          dual_sys->nDualVar_);

  if (cntIter == 1) {
   gradBasedDamp = gradNorm/dual_sys->gradDampFactor_; //earlier, it was 3 ####
  }

  if ((dual_sys->annealIval_ != -1) 
      && (gradNorm < gradBasedDamp) 
      && (dual_sys->tau_ < dual_sys->tauMax_)) {
   std::cout<<"Annealing: grad norm "<<gradNorm
            <<" grad based damp "<<gradBasedDamp<<std::endl;
   cntIterTau = 1;
   dual_sys->tau_ *= dual_sys->tauScale_;
   //gradDampFlag = true;

   //updateMSD(cntIter);
   updateStar(dual_sys, cntIter);

   dual_sys->popGradEnergy();

   gradNorm = myUtils::norm<double>(dual_sys->gradient_, dual_sys->nDualVar_);
   gradBasedDamp = gradNorm/dual_sys->gradDampFactor_;

   if (gradBasedDamp < dual_sys->dampTol_) {
    gradBasedDamp = dual_sys->dampTol_;
   }
  }

  int gradMaxInd = myUtils::argmaxAbs<double>(
                            dual_sys->gradient_,0,dual_sys->nDualVar_);

  double gradMax = std::abs(dual_sys->gradient_[gradMaxInd]);

  int checkIter = -1;

  std::string opName;

  if (cntIter == checkIter) {
   opName = "gradient_" + std::to_string(checkIter) + ".txt";

   std::ofstream opGrad(opName);
   opGrad<<std::scientific;
   opGrad<<std::setprecision(6);
   for (auto grad_elem : dual_sys->gradient_) {
    opGrad<<grad_elem<<std::endl;
   }
   opGrad.close();

   opName = "dual_" + std::to_string(checkIter) + ".txt";

   std::ofstream opDual(opName);
   opDual<<std::scientific;
   opDual<<std::setprecision(6);
   for (auto dual_elem : dual_sys->dualVar_) {
    opDual<<dual_elem<<std::endl;
   }
   opDual.close();
  }

  if ((gradMax < dual_sys->gradTol_)
      || (gradNorm < dual_sys->gradTol_)
      || (dual_sys->pdGap_ < dual_sys->gradTol_)
      || (dual_sys->smallGapIterCnt_ > 10*dual_sys->cntExitCond_)) {
   if ((dual_sys->annealIval_ == -1) || (dual_sys->tau_ == dual_sys->tauMax_)) {
    contIter = false;

    gradNorm = myUtils::norm<double>(dual_sys->gradient_, dual_sys->nDualVar_);
    gradMaxInd = myUtils::argmaxAbs<double>(
                          dual_sys->gradient_,0,dual_sys->nDualVar_);
    gradMax = std::abs(dual_sys->gradient_[gradMaxInd]);
   }
   else  {
    gradBasedDamp = 2*gradNorm;

    if (gradBasedDamp < 0.000001) {
     gradBasedDamp = 0.000001;
    }
   }
  }

  int cntInterval = 10;

  if (((dual_sys->annealIval_ == -1)
        || (dual_sys->tau_ == dual_sys->tauMax_))
      && (cntIter % dual_sys->cntExitCond_ == 0)) {
   dual_sys->recoverFracPrimal();
   dual_sys->recoverFeasPrimal();

   double curNonSmoothPrimalEnergy = dual_sys->compNonSmoothPrimalEnergy();
   double curNonSmoothDualEnergy = dual_sys->compNonSmoothDualEnergy();

   std::cout<<"solveSCD: before update: curNonSmoothDualEnergy "
            <<curNonSmoothDualEnergy<<" curNonSmoothPrimalEnergy "
            <<curNonSmoothPrimalEnergy<<" bestNonSmoothPrimalEnergy "
            <<dual_sys->bestNonSmoothPrimalEnergy_<<std::endl;

   if (dual_sys->pdInitFlag_) {
    dual_sys->bestNonSmoothPrimalEnergy_ = curNonSmoothPrimalEnergy;
    dual_sys->pdInitFlag_ = false;
   }
   else if (curNonSmoothPrimalEnergy > dual_sys->bestNonSmoothPrimalEnergy_) {
    dual_sys->bestNonSmoothPrimalEnergy_ = curNonSmoothPrimalEnergy;
   }

   dual_sys->pdGap_ = (curNonSmoothDualEnergy
                       - dual_sys->bestNonSmoothPrimalEnergy_)
                      /(std::abs(curNonSmoothDualEnergy)
                        + std::abs(dual_sys->bestNonSmoothPrimalEnergy_));

   std::cout<<"solveSCD: after update: curNonSmoothDualEnergy "
            <<curNonSmoothDualEnergy<<" curNonSmoothPrimalEnergy "
            <<curNonSmoothPrimalEnergy<<" bestNonSmoothPrimalEnergy "
            <<dual_sys->bestNonSmoothPrimalEnergy_<<std::endl;
   std::cout<<"solveSCD: PD Gap "<<dual_sys->pdGap_<<std::endl;

   if (dual_sys->pdGap_ < 10*dual_sys->gradTol_) {
    ++dual_sys->smallGapIterCnt_;
   }
  }

#if 1
  if (cntIter % cntInterval == 0) {
   std::cout<<"SolveSCD: ITERATION "<<cntIter
            <<". Tau "<<dual_sys->tau_
            <<" and gradient threshold "<<gradBasedDamp<<".";
   std::cout<<" Current iteration took "
            <<(myUtils::getTime() - tFull)<<" seconds.";
   std::cout<<" Gradient: l-infinity: "<<gradMax<<", Euclidean: "<<gradNorm
            <<". Energy: "<<dual_sys->curEnergy_;
   std::cout<<std::endl;

   double tEnergy = myUtils::getTime();

   std::cout<<std::fixed;
   std::cout<<std::setprecision(6);

   dual_sys->recoverNodeFracPrimal();
   dual_sys->recoverMaxPrimal(dual_sys->primalFrac_);

   double curIntPrimalEnergy = dual_sys->compIntPrimalEnergy();

   if (bestPrimalEnergy < curIntPrimalEnergy) {
    bestPrimalEnergy = curIntPrimalEnergy;
    dual_sys->bestPrimalMax_ = dual_sys->primalMax_;
    dual_sys->timeToBestPrimal_ = myUtils::getTime() - dual_sys->timeStart_;
    std::cout<<"Best primal updated in iteration "<<cntIter
             <<" at time "<<dual_sys->timeToBestPrimal_<<std::endl;
   }
   else if (cntIter == cntInterval) {
    bestPrimalEnergy = curIntPrimalEnergy;
    dual_sys->bestPrimalMax_ = dual_sys->primalMax_;
    dual_sys->timeToBestPrimal_ = myUtils::getTime() - dual_sys->timeStart_;
   }

   std::cout<<" Best integral primal energy "<<bestPrimalEnergy;

   std::cout<<" Non-smooth dual energy ";

   if (dual_sys->sparseFlag_) {
    std::cout<<"Not efficiently computable yet!";
   }
   else {
    std::cout<<dual_sys->compNonSmoothDualEnergy();
   }

   std::cout<<std::endl;

   std::cout<<"Computing energies took "
            <<myUtils::getTime() - tEnergy<<" seconds."<<std::endl;

   tCompEnergies += myUtils::getTime() - tEnergy;
  }

#endif
 } //while ((contIter) && (cntIter < maxIter_))

 std::cout<<"solveSCD: total time taken "
          <<myUtils::getTime() - dual_sys->timeStart_<<" seconds."<<std::endl;

 dual_sys->recoverNodeFracPrimal();
 dual_sys->recoverMaxPrimal(dual_sys->primalFrac_);

 std::cout<<std::fixed;
 std::cout<<std::setprecision(6);

 std::cout<<"solveSCD: Smooth dual energy: ";
 if (dual_sys->sparseFlag_) {
  std::cout<<dual_sys->compEnergySparse(dual_sys->dualVar_)<<std::endl;
 }
 else {
  std::cout<<dual_sys->compEnergy(dual_sys->dualVar_)<<std::endl;
 }
 std::cout<<"solveSCD: Integral primal energy: "
          <<dual_sys->compIntPrimalEnergy()<<std::endl;

 dual_sys->recoverNodeFracPrimal();
 dual_sys->recoverMaxPrimal(dual_sys->primalFrac_);

 std::cout<<"solveSCD: Best integral primal energy: "
          <<bestPrimalEnergy<<std::endl;
 std::cout<<"solveSCD: Non-smooth dual energy: "
          <<dual_sys->compNonSmoothDualEnergy()<<std::endl;
 dual_sys->recoverFracPrimal();
 dual_sys->recoverFeasPrimal();
 std::cout<<"solveSCD: Non-smooth primal energy: "
          <<dual_sys->compNonSmoothPrimalEnergy()<<std::endl;
 std::cout<<"Total time spent on computing energies in each iteration: "
          <<tCompEnergies<<std::endl;

 return 0;
} //solve()

int SolveSCD::updateMSD(DualSys* dual_sys, int cntIter) {

 std::list<int> unprocessCliq(dual_sys->nCliq_);
 std::vector<std::list<int> > unprocessMemNode(dual_sys->nCliq_);

 int cliqCnt = 0;

 for (auto& iCliq : unprocessCliq) {
  iCliq = cliqCnt;

  int sizCliq = dual_sys->subProb_[iCliq].sizCliq_;

  unprocessMemNode[iCliq].resize(sizCliq);

  int nodeCnt = 0;

  for (auto& iNode : unprocessMemNode[iCliq]) {
   iNode = nodeCnt;
   ++nodeCnt;
  }

  ++cliqCnt;
 }

 while (unprocessCliq.size() > 0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> disCliq(1, unprocessCliq.size());

  std::list<int>::iterator iCliq = unprocessCliq.begin();

  int randCliqInd = disCliq(gen) - 1;

  std::advance(iCliq, randCliqInd);

  if (unprocessMemNode[*iCliq].size() == 0) {
   unprocessCliq.erase(iCliq);
  }
  else {
   int cliqOffset = dual_sys->subProb_[*iCliq].getCliqOffset();
   std::vector<int> memNode = dual_sys->subProb_[*iCliq].memNode_;
   std::vector<int> nodeOffset = dual_sys->subProb_[*iCliq].getNodeOffset();

   int nCliqLab = dual_sys->subProb_[*iCliq].nCliqLab_;
   int sizCliq = dual_sys->subProb_[*iCliq].sizCliq_;
   std::vector<short> nodeLabels = dual_sys->subProb_[*iCliq].getNodeLabel();

   std::vector<double> dualVar = dual_sys->subProb_[*iCliq].getDualVar();

   int subDualSiz = dualVar.size();

   std::vector<double> nodeLabSum(subDualSiz);

   std::uniform_int_distribution<> disNode(1, unprocessMemNode[*iCliq].size());

   std::list<int>::iterator iNode = unprocessMemNode[*iCliq].begin();

   int randNodeInd = disNode(gen) - 1;

   std::advance(iNode, randNodeInd);

   int curNode = memNode[*iNode];

   double expVal;

   int nLabel = dual_sys->nLabel_[curNode];

   double f2DenExpMax;
   std::vector<double> f2LabelSum(nLabel);
   std::vector<double> f2DenExp(nLabel);

   double f2Den = 0;

//  #pragma omp parallel for
   for (int i = 0; i != nLabel; ++i) {
    double cliqSumLabel = 0;

//   #pragma omp for reduction(+ : cliqSumLabel)
    for (auto j : dual_sys->cliqPerNode_[curNode]) {
     int nodeInd = myUtils::findNodeIndex(dual_sys->subProb_[j].memNode_, curNode);

     std::vector<double> dualVar = dual_sys->subProb_[j].getDualVar();

     cliqSumLabel += dualVar[
                         dual_sys->subProb_[j].getNodeOffset()[nodeInd] + i];
    }
    f2DenExp[i] = dual_sys->uEnergy_[dual_sys->unaryOffset_[curNode] + i]
                      + cliqSumLabel;
   } //for i = [0:nLabel)

   f2DenExpMax = *std::max_element(f2DenExp.begin(), f2DenExp.end());

   for (int i = 0; i != nLabel; ++i) {
    expVal = exp(dual_sys->tau_*(f2DenExp[i] - f2DenExpMax));
    myUtils::checkRangeError(expVal);

    f2LabelSum[i] = expVal;

    f2Den += expVal;
   }

   for (int i = 0; i != nLabel; ++i) {
    f2LabelSum[i] /= f2Den;
   }

   if (!dual_sys->sparseFlag_) {
    double expVal;

    double f1Den = 0;

    std::vector<double> f1NodeSum(nCliqLab);
    std::vector<double> f1DenExp(nCliqLab);

    double f1DenExpMax = 0;

    for (int i = 0; i != nCliqLab; ++i) {
     double nodeSum = 0;

     double labPull = nCliqLab;

     for (int j = 0; j != sizCliq; ++j) {
      short cliqLabCur;
      labPull /= nodeLabels[j];

      int labPullTwo = ceil((i+1)/labPull);

      if (labPullTwo % nodeLabels[j] == 0) {
       cliqLabCur = nodeLabels[j] - 1;
      }
      else {
       cliqLabCur = (labPullTwo % nodeLabels[j]) - 1;
      }

      nodeSum += dualVar[nodeOffset[j] + cliqLabCur];
     }

     f1NodeSum[i] = nodeSum;

     f1DenExp[i] = dual_sys->subProb_[*iCliq].getCE(i) - nodeSum;

     if (i == 0) {
      f1DenExpMax = f1DenExp[i];
     }
     else {
      if (f1DenExp[i] > f1DenExpMax) {
       f1DenExpMax = f1DenExp[i];
      }
     }
    }

    for (int i = 0; i != nCliqLab; ++i) {
     expVal = exp(dual_sys->tau_*(f1DenExp[i] - f1DenExpMax));
     myUtils::checkRangeError(expVal);
     f1Den += expVal;
     std::vector<short> cliqLab;

     double labPull = nCliqLab;

     for (int j = 0; j != sizCliq; ++j) {
      labPull /= nodeLabels[j];

      int labPullTwo = ceil((i+1)/labPull);

      if (labPullTwo % nodeLabels[j] == 0) {
       cliqLab.push_back(nodeLabels[j] - 1);
      }
      else {
       cliqLab.push_back((labPullTwo % nodeLabels[j]) - 1);
      }

      nodeLabSum[nodeOffset[j] + cliqLab[j]] += expVal;
     }
    }

    for (auto& nodeLab : nodeLabSum) {
      nodeLab /= f1Den;
    }
   }
   else {
    cliqSPSparseSCD(&dual_sys->subProb_[*iCliq],
                    dual_sys->tau_, *iNode, nodeLabSum);
   }

   for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
    int margInd = nodeOffset[*iNode] + iLabel;
    int updateInd = cliqOffset + margInd;

    double logVal,logArg;

//    if ((nodeLabSum[margInd] <= mcPrec_) && (f2LabelSum[iLabel] <= mcPrec_)) {
//     logVal = 0;
//    }
//    else {
//     logVal = log(nodeLabSum[margInd]/f2LabelSum[iLabel]);
//    }

    logArg = nodeLabSum[margInd];
    if (logArg < dual_sys->minNumAllow_) {
     logArg = dual_sys->minNumAllow_;
    }

    logVal = log(logArg);
    myUtils::checkLogError(logVal);

    dual_sys->dualVar_[updateInd] += (1/(2*dual_sys->tau_))*logVal;

    logArg = f2LabelSum[iLabel];
    if (logArg < dual_sys->minNumAllow_) {
     logArg = dual_sys->minNumAllow_;
    }

    logVal = log(logArg);
    myUtils::checkLogError(logVal);

    dual_sys->dualVar_[updateInd] -= (1/(2*dual_sys->tau_))*logVal;
   }

   unprocessMemNode[*iCliq].erase(iNode);

   dual_sys->distributeDualVars();
  }
 }

 return 0;
}

int SolveSCD::updateStar(DualSys* dual_sys, int cntIter) {

 std::list<int> unprocessNode(dual_sys->nNode_);

 std::iota(unprocessNode.begin(),unprocessNode.end(),0);

 while (unprocessNode.size() > 0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> disNode(1, unprocessNode.size());

  std::list<int>::iterator iNode = unprocessNode.begin();

  int randNodeInd = disNode(gen) - 1;

  std::advance(iNode, randNodeInd);

  int curNode = *iNode;

  int nLabel = dual_sys->nLabel_[curNode];

  double expVal;

  double f2DenExpMax;
  std::vector<double> f2LabelSum(nLabel);
  std::vector<double> f2DenExp(nLabel);

  double f2Den = 0;

//  #pragma omp parallel for
  for (int i = 0; i != nLabel; ++i) {
   double cliqSumLabel = 0;

//   #pragma omp for reduction(+ : cliqSumLabel)
   for (auto j : dual_sys->cliqPerNode_[curNode]) {
    int nodeInd = myUtils::findNodeIndex(
                               dual_sys->subProb_[j].memNode_, curNode);

    std::vector<double> dualVar = dual_sys->subProb_[j].getDualVar();

    cliqSumLabel += dualVar[dual_sys->subProb_[j].getNodeOffset()[nodeInd] + i];
   }
   f2DenExp[i] = dual_sys->uEnergy_[dual_sys->unaryOffset_[curNode] + i]
                     + cliqSumLabel;
  } //for i = [0:nLabel)

  f2DenExpMax = *std::max_element(f2DenExp.begin(), f2DenExp.end());

  for (int i = 0; i != nLabel; ++i) {
   expVal = exp(dual_sys->tau_*(f2DenExp[i] - f2DenExpMax));
   myUtils::checkRangeError(expVal);

   f2LabelSum[i] = expVal;

   f2Den += expVal;
  }

  for (int i = 0; i != nLabel; ++i) {
   f2LabelSum[i] /= f2Den;
  }

  std::vector<std::vector<double> >
       nodeLabSumSet(dual_sys->cliqPerNode_[curNode].size());

  std::vector<double> nodeLabSumLogProd(nLabel);

  int nCliq = 0;
  double cliqPerNode = 
       static_cast<double>(dual_sys->cliqPerNode_[curNode].size());

  for (auto iCliq : dual_sys->cliqPerNode_[curNode]) {
   std::vector<int> memNode = dual_sys->subProb_[iCliq].memNode_;
   std::vector<int> nodeOffset = dual_sys->subProb_[iCliq].getNodeOffset();

   int nodeInd = myUtils::findNodeIndex(memNode, curNode);

   int nCliqLab = dual_sys->subProb_[iCliq].nCliqLab_;
   int sizCliq = dual_sys->subProb_[iCliq].sizCliq_;
   std::vector<short> nodeLabels = dual_sys->subProb_[iCliq].getNodeLabel();

   std::vector<double> dualVar = dual_sys->subProb_[iCliq].getDualVar();

   int subDualSiz = dualVar.size();

   std::vector<double> nodeLabSum(subDualSiz);

   if (!dual_sys->sparseFlag_) {
    double expVal;

    double f1Den = 0;

    std::vector<double> f1NodeSum(nCliqLab);
    std::vector<double> f1DenExp(nCliqLab);

    double f1DenExpMax = 0;

    for (int i = 0; i != nCliqLab; ++i) {
     double nodeSum = 0;

     double labPull = nCliqLab;

     for (int j = 0; j != sizCliq; ++j) {
      short cliqLabCur;
      labPull /= nodeLabels[j];

      int labPullTwo = ceil((i+1)/labPull);

      if (labPullTwo % nodeLabels[j] == 0) {
       cliqLabCur = nodeLabels[j] - 1;
      }
      else {
       cliqLabCur = (labPullTwo % nodeLabels[j]) - 1;
      }

      nodeSum += dualVar[nodeOffset[j] + cliqLabCur];
     }

     f1NodeSum[i] = nodeSum;

     f1DenExp[i] = dual_sys->subProb_[iCliq].getCE(i) - nodeSum;

     if (i == 0) {
      f1DenExpMax = f1DenExp[i];
     }
     else {
      if (f1DenExp[i] > f1DenExpMax) {
       f1DenExpMax = f1DenExp[i];
      }
     }
    }

    for (int i = 0; i != nCliqLab; ++i) {
     expVal = exp(dual_sys->tau_*(f1DenExp[i] - f1DenExpMax));
     myUtils::checkRangeError(expVal);
     f1Den += expVal;
     std::vector<short> cliqLab;

     double labPull = nCliqLab;

     for (int j = 0; j != sizCliq; ++j) {
      labPull /= nodeLabels[j];

      int labPullTwo = ceil((i+1)/labPull);

      if (labPullTwo % nodeLabels[j] == 0) {
       cliqLab.push_back(nodeLabels[j] - 1);
      }
      else {
       cliqLab.push_back((labPullTwo % nodeLabels[j]) - 1);
      }

      nodeLabSum[nodeOffset[j] + cliqLab[j]] += expVal;
     }
    }

    for (auto& nodeLab : nodeLabSum) {
     nodeLab /= f1Den;
    }
   }
   else {
    cliqSPSparseSCD(&dual_sys->subProb_[iCliq],
                    dual_sys->tau_, nodeInd, nodeLabSum);
   }

   nodeLabSumSet[nCliq] = nodeLabSum;

   for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
   int margInd = nodeOffset[nodeInd] + iLabel;

    double logArg = nodeLabSum[margInd];

    if (logArg < dual_sys->minNumAllow_) {
     logArg = dual_sys->minNumAllow_;
    }

    double logVal = log(logArg);
    myUtils::checkLogError(logVal);

    nodeLabSumLogProd[iLabel] += logVal;
   }

   ++nCliq;
  } //for iCliq

  nCliq = 0;

 for (auto iCliq : dual_sys->cliqPerNode_[curNode]) {
   int cliqOffset = dual_sys->subProb_[iCliq].getCliqOffset();
   std::vector<int> memNode = dual_sys->subProb_[iCliq].memNode_;
   std::vector<int> nodeOffset = dual_sys->subProb_[iCliq].getNodeOffset();

   int nodeInd = myUtils::findNodeIndex(memNode, curNode);

   for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
    int margInd = nodeOffset[nodeInd] + iLabel;
    int updateInd = cliqOffset + margInd;

    double logArg = f2LabelSum[iLabel];

    if (logArg < dual_sys->minNumAllow_) {
     logArg = dual_sys->minNumAllow_;
    }

    double logVal = log(logArg);
    myUtils::checkLogError(logVal);

    //std::cout<<"updateStar: nodeLabSumLogProd: "<<nodeLabSumLogProd[iLabel]<<" log(f2LabelSum[iLabel]) "<<logVal<<" update value one "<<(-1/(cliqPerNode_[curNode].size()+1))*(1/tau_)*(logVal + nodeLabSumLogProd[iLabel]);

    dual_sys->dualVar_[updateInd] -= (1/(cliqPerNode+1))
                                     *(1/dual_sys->tau_)
                                     *(logVal + nodeLabSumLogProd[iLabel]);

    logArg = nodeLabSumSet[nCliq][margInd];

    if (logArg < dual_sys->minNumAllow_) {
     logArg = dual_sys->minNumAllow_;
    }

    logVal = log(logArg);
    myUtils::checkLogError(logVal);

    dual_sys->dualVar_[updateInd] += (1/dual_sys->tau_)*logVal;

    //std::cout<<" value two "<<(1/tau_)*logVal<<std::endl;
   }

   ++nCliq;
  }

  dual_sys->distributeDualVars();

  unprocessNode.erase(iNode);
 } //while loop

 return 0;
}

