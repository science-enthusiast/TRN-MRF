//note: within a clique if the nodes are ordered as 0,1,2,..., 0 is the most significant node for calculating stride.

#include <vector>
#include "ICFS.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <cerrno>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <random>
#include <limits>
#include <numeric>
#include "dualSys.hpp"
#include "myUtils.hpp"
#include "hessVecMult.hpp"
#include "quasiNewton.hpp"
#include "cliqSPSparseHessian.hpp"
#include "cliqSPSparseFista.hpp"
#include "cliqSPSparseEnergy.hpp"
#include "cliqSPSparseGradEnergy.hpp"
#include "cliqSPSparseSCD.hpp"
#include "polyInterp.h"
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/src/IterativeSolvers/IncompleteCholesky.h>
#include <Eigen/src/Core/util/Constants.h>

int debugEigenVec(Eigen::VectorXd);

dualSys::dualSys(int nNode, std::vector<short> nLabel, double tau, double stgCvxCoeff, int mIter, int annealIval, bool stgCvxFlag): nDualVar_(0), nNode_(nNode), numLabTot_(0), nLabel_(nLabel), nCliq_(0), tau_(tau), tauStep_(tau), solnQual_(false), finalEnergy_(0), stgCvxFlag_(stgCvxFlag), stgCvxCoeff_(stgCvxCoeff), maxIter_(mIter), lsAlpha_(1), annealIval_(annealIval) {

 for (int i = 0; i != nNode_; ++i) {
  unaryOffset_.push_back(numLabTot_);
  numLabTot_ += nLabel_[i];
 }

 uEnergy_.resize(numLabTot_);
 cliqPerNode_.resize(nNode_);
 primalFrac_.resize(numLabTot_);
}

int  dualSys::addNode(int n, std::vector<double> uEnergy) {
 for (int l = 0; l != nLabel_[n]; ++l) {
  uEnergy_[unaryOffset_[n] + l] = uEnergy[l];
 }

 return 0;
}

int dualSys::addCliq(const std::vector<int> & nodeList, std::vector<double> *cEnergy) {
 sparseFlag_ = false;

 std::vector<short> labelList;

 for (std::vector<int>::const_iterator i = nodeList.begin(); i != nodeList.end(); ++i) {
  cliqPerNode_[*i].push_back(nCliq_);
  labelList.push_back(nLabel_[*i]);
 }

 subProb_.push_back(subProblem(nodeList, labelList, cEnergy));

 //cEnergy2DVec_.push_back(*cEnergy); //2D vector of all clique energies
 //cliqOffset_.push_back(cEnergy2DVec_.size() - 1);

 return ++nCliq_;
}

int dualSys::addCliq(const std::vector<int> & nodeList, std::vector<std::pair<std::vector<short>,double> > *cEnergy) {
 sparseFlag_ = true;

 std::vector<short> labelList;

 for (std::vector<int>::const_iterator i = nodeList.begin(); i != nodeList.end(); ++i) {
  cliqPerNode_[*i].push_back(nCliq_);
  labelList.push_back(nLabel_[*i]);
 }

 subProb_.push_back(subProblem(nodeList, labelList, cEnergy));

 return ++nCliq_;
}

int dualSys::addCliq(const std::vector<int> & nodeList, const std::vector<double> & pixVals) {
 std::vector<short> labelList;

 for (std::vector<int>::const_iterator i = nodeList.begin(); i != nodeList.end(); ++i) {
  cliqPerNode_[*i].push_back(nCliq_);
  labelList.push_back(nLabel_[*i]);
 }

 subProb_.push_back(subProblem(nodeList,labelList, pixVals));

 return ++nCliq_;
}

int dualSys::addCliq(const std::vector<int> & nodeList, int rowSiz, int colSiz, double *sparseKappa, std::map<int,double> *sparseEnergy, std::set<int> *sparseIndex) {
 sparseFlag_ = true;

 std::vector<short> labelList;

 for (std::vector<int>::const_iterator i = nodeList.begin(); i != nodeList.end(); ++i) {
  cliqPerNode_[*i].push_back(nCliq_);
  labelList.push_back(nLabel_[*i]);
 }

 subProb_.push_back(subProblem(nodeList, labelList, rowSiz, colSiz, sparseKappa, sparseEnergy, sparseIndex));

 return ++nCliq_;
}

int dualSys::prepareDualSys() {

 maxCliqSiz_ = 0;
 maxNumNeigh_ = 0;
 nDualVar_ = 0;

 pdGap_ = 1;
 pdInitFlag_ = true;
 smallGapIterCnt_ = 0;

 std::cout<<"Strong convexity flag "<<stgCvxFlag_<<std::endl;

 nLabelMax_ = *std::max_element(nLabel_.begin(), nLabel_.end());

 nLabCom_ = nLabelMax_;

 nSubDualSizCom_ = subProb_[0].getDualSiz();

 std::vector<int> reserveVec; //::Constant(nDualVar_,maxNumNeigh_*maxCliqSiz_*nLabelMax_);

 for (int iNode = 0; iNode != nNode_; ++iNode) {
  std::sort(cliqPerNode_[iNode].begin(),cliqPerNode_[iNode].end());
 }

 for (int iCliq = 0; iCliq != nCliq_; ++iCliq) {
  subProb_[iCliq].setCliqOffset(nDualVar_);

  if (subProb_[iCliq].sizCliq_ > maxCliqSiz_) {
   maxCliqSiz_ = subProb_[iCliq].sizCliq_;
  }

  nDualVar_ += subProb_[iCliq].getDualVar().size();

  for (std::vector<int>::iterator j = subProb_[iCliq].memNode_.begin(); j != subProb_[iCliq].memNode_.end(); ++j) {
   //subProb_[i].cliqNeigh_.insert(cliqPerNode_[*j].begin(), cliqPerNode_[*j].end());
   for (std::vector<int>::iterator iNeigh = cliqPerNode_[*j].begin(); iNeigh != cliqPerNode_[*j].end(); ++iNeigh) {
    if (*iNeigh >= iCliq) {
     subProb_[iCliq].cliqNeigh_.insert(*iNeigh);
    }
   }
  }
 } //for iCliq

 blkJacob_.resize(nCliq_);
 blkDiag_.resize(nCliq_);

 unsigned long long elemCnt = 0; //####

 reserveCliqBlk_.resize(nDualVar_);

 std::cout<<"Clique neighbours";

 for (int i = 0; i != nCliq_; ++i) {
  std::cout<<" "<<subProb_[i].cliqNeigh_.size();

  if (subProb_[i].cliqNeigh_.size() == 1) {
   std::cout<<" clique "<<i<<" ( ";

   for (std::set<int>::iterator iCliq = subProb_[i].cliqNeigh_.begin(); iCliq != subProb_[i].cliqNeigh_.end(); ++iCliq) {
    std::cout<<*iCliq<<" ";
   }

   std::cout<<") ";
  }

  if (subProb_[i].cliqNeigh_.size() > maxNumNeigh_) {
   maxNumNeigh_ = subProb_[i].cliqNeigh_.size();
  }

  int reserveSiz = 0;

  for (std::set<int>::iterator n = subProb_[i].cliqNeigh_.begin(); n != subProb_[i].cliqNeigh_.end(); ++n) {
   if (*n >= i) {
    if (i == *n) {
     reserveSiz += subProb_[*n].getDualVar().size();
     elemCnt += subProb_[*n].getDualVar().size();
    }
    else {
     reserveSiz += subProb_[*n].getNodeLabel()[0]; //$$$ needs to be max. label size for best performance
     elemCnt += subProb_[*n].getNodeLabel()[0];
    }
   }
  }

  std::cout<<" <"<<reserveSiz<<"> ";

  for (std::size_t j = 0; j != subProb_[i].getDualVar().size(); ++j) {
   reserveVec.push_back(reserveSiz);

   reserveCliqBlk_[subProb_[i].getCliqOffset() + j] = subProb_[i].getDualSiz();

   elemCnt += reserveSiz;
  }
 }

 std::cout<<std::endl;

 std::cout<<"Maximum number of neighbours for a clique "<<maxNumNeigh_<<std::endl;

 std::cout<<"non-zero reserved size for hessian "<<elemCnt<<std::endl;

 cHess_ = new double[elemCnt];

 reserveHess_.resize(nDualVar_);

 unsigned long long nonZeroTerms = 0;

 for (std::size_t i = 0; i != nDualVar_; ++i) {
  reserveHess_[i] = reserveVec[i];
  nonZeroTerms += reserveVec[i];
 }

// std::cout<<"non-zero terms "<<nonZeroTerms<<std::endl;

 std::cout<<"no. of dual variables "<<nDualVar_<<std::endl;
 std::cout<<"hessian size "<<nDualVar_*nDualVar_<<std::endl;
// std::cout<<"maxNumNeigh_"<<maxNumNeigh_<<std::endl;
// std::cout<<"maxCliqSiz_"<<maxCliqSiz_<<std::endl;
// std::cout<<"nLabelMax_"<<nLabelMax_<<std::endl;

 primalMax_.resize(nNode_);

 dualVar_.resize(nDualVar_);

 gradient_.resize(nDualVar_);

 eigenHess_.resize(nDualVar_, nDualVar_);
 eigenGrad_.resize(nDualVar_);

 cliqBlkHess_.resize(nDualVar_, nDualVar_);

 nodeBlkHess_.resize(nNode_);
 //nodeBlk_.resize(nNode_);
 reserveNodeBlk_.resize(nNode_);

 for (int iNode = 0; iNode != nNode_; ++iNode) {
  nodeBlkHess_[iNode].resize(nLabCom_, nLabCom_);
  //nodeBlk_[iNode] = new double[nLabCom_*(nLabCom_+1)/2];
  //reserveNodeBlk_[iNode].resize(nDualVar_);
 }

 sparseNodeOffset_.resize(nNode_);

 for (int iCliq = 0; iCliq != nCliq_; ++iCliq) {
  std::vector<int> memNode = subProb_[iCliq].getMemNode();
  int cliqOffset = subProb_[iCliq].getCliqOffset();
  std::vector<int> nodeOffset = subProb_[iCliq].getNodeOffset();
  for (int iMemNode = 0; iMemNode != subProb_[iCliq].getCliqSiz(); ++iMemNode) {
   //for (int iLabel = 0; iLabel != nLabel_[memNode[iMemNode]]; ++iLabel) {
    //reserveNodeBlk_[memNode[iMemNode]][cliqOffset + nodeOffset[iMemNode] + iLabel] = nLabCom_;
    sparseNodeOffset_[memNode[iMemNode]].push_back(cliqOffset + nodeOffset[iMemNode]);
   //}
  }
 }

 newtonStep_.resize(nDualVar_);

 //inc. Cholesky preconditioner
 hessDiag_ = new double[nDualVar_];
 hessColPtr_ = new int[nDualVar_+1];
 nNonZeroLower_ = 0;

// assignPrimalVars("/home/hari/libraries/drwn-1.9.0/bin/testLabels.txt"); //assign ground truth labeling
// std::cout<<"energy (hard-coded assignment): "<<compIntPrimalEnergy()<<std::endl;

// recoverFracPrimal();
// recoverMaxPrimal(primalFrac_);

// std::cout<<"Initial integral primal energy: "<<compIntPrimalEnergy()<<std::endl;
// std::cout<<"Initial non-smooth dual energy: "<<compNonSmoothDualEnergy()<<std::endl;

// std::ifstream dualVarFile("debugDual.txt");

// for (int i = 0; i != nDualVar_; ++i) {
//  dualVarFile>>dualVar_[i];
// }

// distributeDualVars();

// tau_ = 8192;

 return 0;
}

int dualSys::popGradHessEnergyPerf(int cntIter) {
 double tRoutine = myUtils::getTime();

 double f2DenExpMax;
 std::vector<double> f2LabelSum(numLabTot_);

 curEnergy_ = 0;
 double curEnergyRdx = 0;

// double maxElem = 0, minElem = 0;

 //#pragma omp parallel for reduction(+ : curEnergyRdx)
 for (int curNode = 0; curNode < nNode_; ++curNode) {
  double expVal;

  int nLabel = nLabel_[curNode];

  std::vector<double> f2DenExp(nLabel);

  double f2Den = 0;

//  #pragma omp parallel for
  for (int i = 0; i != nLabel; ++i) {
   double cliqSumLabel = 0;

//   #pragma omp for reduction(+ : cliqSumLabel)
   for (std::vector<int>::iterator j = cliqPerNode_[curNode].begin(); j != cliqPerNode_[curNode].end(); ++j) {
    int nodeInd = myUtils::findNodeIndex(subProb_[*j].memNode_, curNode);

    std::vector<double> dualVar = subProb_[*j].getDualVar();

    cliqSumLabel += dualVar[subProb_[*j].getNodeOffset()[nodeInd] + i];
   }
   f2DenExp[i] = uEnergy_[unaryOffset_[curNode] + i] + cliqSumLabel;
  } //for i = [0:nLabel)

  f2DenExpMax = *std::max_element(f2DenExp.begin(), f2DenExp.end());

  for (int i = 0; i != nLabel; ++i) {
   expVal = exp(tau_*(f2DenExp[i] - f2DenExpMax));
   myUtils::checkRangeError(expVal);

   f2LabelSum[unaryOffset_[curNode] + i] = expVal;

   f2Den += expVal;
  }

  for (int i = 0; i != nLabel; ++i) {
   f2LabelSum[unaryOffset_[curNode] + i] /= f2Den;
  }

  curEnergyRdx += (1/tau_)*log(f2Den) + f2DenExpMax;

  //int nodeBlkInd = 0;

  //store node block
  for (int jLabel = 0; jLabel != nLabel; ++jLabel) {
   double f2NumTerm = f2LabelSum[unaryOffset_[curNode] + jLabel];

   //nodeBlkHess_[curNode](jLabel,jLabel) = tau_*f2NumTerm*(1-f2NumTerm);
   //nodeBlk_[curNode][jLabel*(nLabel+1)] = tau_*f2NumTerm*(1-f2NumTerm);
   //++nodeBlkInd;

   for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
    if (iLabel != jLabel) {
     nodeBlkHess_[curNode](iLabel, jLabel) = -1*tau_*f2NumTerm*f2LabelSum[unaryOffset_[curNode] + iLabel];
    }
    else {
     nodeBlkHess_[curNode](jLabel,jLabel) = tau_*f2NumTerm*(1-f2NumTerm);
    }

     //nodeBlk_[curNode][jLabel*nLabel + iLabel] = -1*tau_*f2NumTerm*f2LabelSum[unaryOffset_[curNode] + iLabel];
    //}
    //++nodeBlkInd;
   } //for iLabel
  } //for jLabel
  //store node block
 } //for curNode

 double f1Den = 0;

 //#pragma omp parallel for reduction(+ : curEnergyRdx)
 for (int cliqJ = 0; cliqJ < nCliq_; ++cliqJ) {

  int nCliqLab = subProb_[cliqJ].nCliqLab_;
  int sizCliqJ = subProb_[cliqJ].sizCliq_;
  std::vector<short> nLabelJ = subProb_[cliqJ].getNodeLabel();

  std::vector<double> dualVar = subProb_[cliqJ].getDualVar();
  std::vector<int> nodeOffsetJ = subProb_[cliqJ].getNodeOffset();
  int cliqOffsetJ = subProb_[cliqJ].getCliqOffset();
  std::set<int> cliqNeigh = subProb_[cliqJ].cliqNeigh_;
  std::vector<int> memNodeJ = subProb_[cliqJ].memNode_;
  std::vector<double> pixValJ = subProb_[cliqJ].getPixVal();

  int subDualSiz = dualVar.size();

  std::vector<double> nodeLabSum(subDualSiz);
  std::vector<double> nodePairLabSum(subDualSiz*subDualSiz);

  Eigen::MatrixXd cliqBlk(subDualSiz,subDualSiz);
  std::vector<double> diagReg(subDualSiz);

  int blkCol = 0;

  if (!sparseFlag_) {
   double expVal;

   f1Den = 0;

   std::vector<double> f1NodeSum(nCliqLab);
   std::vector<double> f1DenExp(nCliqLab);

   double f1DenExpMax = 0;

   for (int i = 0; i != nCliqLab; ++i) {
    double nodeSum = 0;

    double labPull = nCliqLab;

    for (int j = 0; j != sizCliqJ; ++j) {
     short cliqLabCur;
     labPull /= nLabelJ[j];

     int labPullTwo = ceil((i+1)/labPull);

     if (labPullTwo % nLabelJ[j] == 0) {
      cliqLabCur = nLabelJ[j] - 1;
     }
     else {
      cliqLabCur = (labPullTwo % nLabelJ[j]) - 1;
     }

     nodeSum += dualVar[nodeOffsetJ[j] + cliqLabCur];
    }

    f1NodeSum[i] = nodeSum;

    f1DenExp[i] = subProb_[cliqJ].getCE(i) - nodeSum;

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
    expVal = exp(tau_*(f1DenExp[i] - f1DenExpMax));
    myUtils::checkRangeError(expVal);
    f1Den += expVal;
    std::vector<short> cliqLab;

    double labPull = nCliqLab;

    for (int j = 0; j != sizCliqJ; ++j) {
     labPull /= nLabelJ[j];

     int labPullTwo = ceil((i+1)/labPull);

     if (labPullTwo % nLabelJ[j] == 0) {
      cliqLab.push_back(nLabelJ[j] - 1);
     }
     else {
      cliqLab.push_back((labPullTwo % nLabelJ[j]) - 1);
     }

     nodeLabSum[nodeOffsetJ[j] + cliqLab[j]] += expVal;

     for (int k = 0; k != j; ++k) {
      nodePairLabSum[(nodeOffsetJ[k] + cliqLab[k])*subDualSiz + nodeOffsetJ[j] + cliqLab[j]] += expVal;
     }
    }
   }

   for (std::vector<double>::iterator nodeLabIter = nodeLabSum.begin(); nodeLabIter != nodeLabSum.end(); ++nodeLabIter) {
    *nodeLabIter /= f1Den;
   }

   for (std::vector<double>::iterator nodePairLabIter = nodePairLabSum.begin(); nodePairLabIter != nodePairLabSum.end(); ++nodePairLabIter) {
    *nodePairLabIter /= f1Den;
   }

   curEnergyRdx += (1/tau_)*log(f1Den) + f1DenExpMax;
  }
  else {
   double subEnergy = 0;

   cliqSPSparseHessian(&subProb_[cliqJ], tau_, subEnergy, nodeLabSum, nodePairLabSum);
   //cliqSPSparseGradEnergy(&subProb_[cliqJ], dualVar, tau_, subEnergy, nodeLabSum);

   curEnergyRdx += subEnergy;
  }

  for (int elemJ = 0; elemJ != sizCliqJ; ++elemJ) {
   for (int labelJ = 0; labelJ != nLabelJ[elemJ]; ++labelJ) {
    int curJ = cliqOffsetJ + nodeOffsetJ[elemJ] + labelJ;

    int curJNode = memNodeJ[elemJ];

    gradient_[curJ] = -1*nodeLabSum[nodeOffsetJ[elemJ] + labelJ] + f2LabelSum[unaryOffset_[curJNode] + labelJ];

    if (isnan(gradient_[curJ])) {
     std::cout<<"GRADIENT ENTRY IS NAN!"<<std::endl;
     //return -1;
    }

    double* f2Arr = new double[nLabelJ[elemJ]];

    double f2NumTermJ = f2LabelSum[unaryOffset_[curJNode] + labelJ];

    for (int labelI = 0; labelI != nLabelJ[elemJ]; ++labelI) {
     if (labelI == labelJ) {
      f2Arr[labelI] = tau_*f2NumTermJ*(1 - f2NumTermJ);
     }
     else {
      f2Arr[labelI] = -1*tau_*f2NumTermJ*f2LabelSum[unaryOffset_[curJNode] + labelI];
     }
    }

    int blkRow = 0;

    for (int elemI = 0; elemI != sizCliqJ; ++elemI) {
     for (int labelI = 0; labelI != nLabelJ[elemI]; ++labelI) {

      double f1 = 0, f2 = 0;

      double nIlISumF1 = nodeLabSum[nodeOffsetJ[elemI] + labelI];
      double nJlJSumF1 = nodeLabSum[nodeOffsetJ[elemJ] + labelJ];
      double nInJlIlJSumF1;

      if ((elemI == elemJ) && (labelI == labelJ)) {
       f1 = nIlISumF1*(1 -nIlISumF1);
      }
      else if ((elemI == elemJ) && (labelI != labelJ)) {
       f1 = -1*nIlISumF1*nJlJSumF1;
      }
      else {
       if (elemI < elemJ) {
        nInJlIlJSumF1 = nodePairLabSum[(nodeOffsetJ[elemI] + labelI)*subDualSiz + nodeOffsetJ[elemJ] + labelJ];
       }
       else {
        nInJlIlJSumF1 = nodePairLabSum[(nodeOffsetJ[elemJ] + labelJ)*subDualSiz + nodeOffsetJ[elemI] + labelI];
       }

       f1 = (nInJlIlJSumF1 - nIlISumF1*nJlJSumF1);
      }

      int curI = cliqOffsetJ + nodeOffsetJ[elemI] + labelI;

      double curVal = 0;

      if (curI == curJ) {
       curVal += dampLambda_;
      }

      if (curJNode == memNodeJ[elemI]) {
       f2 = f2Arr[labelI];
      }

      curVal += tau_*f1 + f2;

      if ((curI == curJ) && (dampLambda_ < 0.001)) {
       diagReg[blkRow] = 0.001;
      }

      cliqBlk(blkRow,blkCol) = curVal;

      ++blkRow;
     } //labelI
    } //elemI

    delete [] f2Arr;

    ++blkCol;
   } // for labelJ
  } //for elemJ

  blkDiag_[cliqJ] = cliqBlk;

  for (int iDiag = 0; iDiag != subDualSiz; ++iDiag) {
   cliqBlk(iDiag, iDiag) += diagReg[iDiag];
  }

  blkJacob_[cliqJ] = cliqBlk.inverse();
 } //for cliqJ

 curEnergy_ = curEnergyRdx;

 //std::cout<<"Hessian matrix: max element "<<maxElem<<" min element "<<minElem<<std::endl;

 if (isnan(curEnergy_)) {
  std::cout<<"ENERGY INDEED NAN!"<<std::endl;
  return -1;
 }

 std::cout<<"popGradHessEnergyPerf took "<<myUtils::getTime()-tRoutine<<" seconds."<<std::endl;
 std::cout<<std::flush;

 return 0;
}

int dualSys::popGradHessEnergy(int cntIter) {

 double f2DenExpMax;
 std::vector<double> f2LabelSum(numLabTot_);

 curEnergy_ = 0;

// #pragma omp parallel for
 for (int curNode = 0; curNode < nNode_; ++curNode) {
  double expVal;

  int nLabel = nLabel_[curNode];

  std::vector<double> f2DenExp(nLabel);

  double f2Den = 0;

//  #pragma omp parallel for
  for (int i = 0; i != nLabel; ++i) {
   double cliqSumLabel = 0;

//   #pragma omp for reduction(+ : cliqSumLabel)
   for (std::vector<int>::iterator j = cliqPerNode_[curNode].begin(); j != cliqPerNode_[curNode].end(); ++j) {
    int nodeInd = myUtils::findNodeIndex(subProb_[*j].memNode_, curNode);

    std::vector<double> dualVar = subProb_[*j].getDualVar();

    cliqSumLabel += dualVar[subProb_[*j].getNodeOffset()[nodeInd] + i];
   }
   f2DenExp[i] = uEnergy_[unaryOffset_[curNode] + i] + cliqSumLabel;
  } //for i = [0:nLabel)

  f2DenExpMax = *std::max_element(f2DenExp.begin(), f2DenExp.end());

  for (int i = 0; i != nLabel; ++i) {
   expVal = exp(tau_*(f2DenExp[i] - f2DenExpMax));
   myUtils::checkRangeError(expVal);

   f2LabelSum[unaryOffset_[curNode] + i] = expVal;

   f2Den += expVal;
  }

  for (int i = 0; i != nLabel; ++i) {
   f2LabelSum[unaryOffset_[curNode] + i] /= f2Den;
  }

  curEnergy_ += (1/tau_)*log(f2Den) + f2DenExpMax;
 } //for curNode

 double f1Den = 0;

// #pragma omp parallel for
 for (int cliqJ = 0; cliqJ < nCliq_; ++cliqJ) {

  int nCliqLab = subProb_[cliqJ].nCliqLab_;
  int sizCliqJ = subProb_[cliqJ].sizCliq_;
  std::vector<short> nLabelJ = subProb_[cliqJ].getNodeLabel();

  std::vector<double> dualVar = subProb_[cliqJ].getDualVar();
  std::vector<int> nodeOffsetJ = subProb_[cliqJ].getNodeOffset();
  int cliqOffsetJ = subProb_[cliqJ].getCliqOffset();
  std::set<int> cliqNeigh = subProb_[cliqJ].cliqNeigh_;
  std::vector<int> memNodeJ = subProb_[cliqJ].memNode_;
  std::vector<double> pixValJ = subProb_[cliqJ].getPixVal();

  int subDualSiz = dualVar.size();

  std::vector<double> nodeLabSum(subDualSiz);
  std::vector<double> nodePairLabSum(subDualSiz*subDualSiz);

  Eigen::MatrixXd cliqBlk(subDualSiz,subDualSiz);
  std::vector<double> diagReg(subDualSiz);

  int blkCol = 0;

  if (!sparseFlag_) {
   double expVal;

   f1Den = 0;

   std::vector<double> f1NodeSum(nCliqLab);
   std::vector<double> f1DenExp(nCliqLab);

   double f1DenExpMax = 0;

   for (int i = 0; i != nCliqLab; ++i) {
    double nodeSum = 0;

    double labPull = nCliqLab;

    for (int j = 0; j != sizCliqJ; ++j) {
     short cliqLabCur;
     labPull /= nLabelJ[j];

     int labPullTwo = ceil((i+1)/labPull);

     if (labPullTwo % nLabelJ[j] == 0) {
      cliqLabCur = nLabelJ[j] - 1;
     }
     else {
      cliqLabCur = (labPullTwo % nLabelJ[j]) - 1;
     }

     nodeSum += dualVar[nodeOffsetJ[j] + cliqLabCur];
    }

    f1NodeSum[i] = nodeSum;

    f1DenExp[i] = subProb_[cliqJ].getCE(i) - nodeSum;

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
    expVal = exp(tau_*(f1DenExp[i] - f1DenExpMax));
    myUtils::checkRangeError(expVal);
    f1Den += expVal;
    std::vector<short> cliqLab;

    double labPull = nCliqLab;

    for (int j = 0; j != sizCliqJ; ++j) {
     labPull /= nLabelJ[j];

     int labPullTwo = ceil((i+1)/labPull);

     if (labPullTwo % nLabelJ[j] == 0) {
      cliqLab.push_back(nLabelJ[j] - 1);
     }
     else {
      cliqLab.push_back((labPullTwo % nLabelJ[j]) - 1);
     }

     nodeLabSum[nodeOffsetJ[j] + cliqLab[j]] += expVal;

     for (int k = 0; k != j; ++k) {
      nodePairLabSum[(nodeOffsetJ[k] + cliqLab[k])*subDualSiz + nodeOffsetJ[j] + cliqLab[j]] += expVal;
     }
    }
   }

   for (std::vector<double>::iterator nodeLabIter = nodeLabSum.begin(); nodeLabIter != nodeLabSum.end(); ++nodeLabIter) {
    *nodeLabIter /= f1Den;
   }

   for (std::vector<double>::iterator nodePairLabIter = nodePairLabSum.begin(); nodePairLabIter != nodePairLabSum.end(); ++nodePairLabIter) {
    *nodePairLabIter /= f1Den;
   }

   curEnergy_ += (1/tau_)*log(f1Den) + f1DenExpMax;
  }
  else {
   double subEnergy = 0;

   cliqSPSparseHessian(&subProb_[cliqJ], tau_, subEnergy, nodeLabSum, nodePairLabSum);

   curEnergy_ += subEnergy;
  }

  for (int elemJ = 0; elemJ != sizCliqJ; ++elemJ) {
   for (int labelJ = 0; labelJ != nLabelJ[elemJ]; ++labelJ) {
    int curJ = cliqOffsetJ + nodeOffsetJ[elemJ] + labelJ;

    int curJNode = memNodeJ[elemJ];

    gradient_[curJ] = -1*nodeLabSum[nodeOffsetJ[elemJ] + labelJ] + f2LabelSum[unaryOffset_[curJNode] + labelJ];

    if (isnan(gradient_[curJ])) {
     std::cout<<"GRADIENT ENTRY IS NAN!"<<std::endl;

     return -1;
    }

    double* f2Arr = new double[nLabelJ[elemJ]];

    double f2NumTermJ = f2LabelSum[unaryOffset_[curJNode] + labelJ];

    for (int labelI = 0; labelI != nLabelJ[elemJ]; ++labelI) {
     if (labelI == labelJ) {
      f2Arr[labelI] = tau_*f2NumTermJ*(1 - f2NumTermJ);
     }
     else {
      f2Arr[labelI] = -1*tau_*f2NumTermJ*f2LabelSum[unaryOffset_[curJNode] + labelI];
     }
    }

    for (std::set<int>::iterator cliqI = cliqNeigh.begin(); cliqI != cliqNeigh.end(); ++cliqI) {

     int sizCliqI = subProb_[*cliqI].sizCliq_; //OPTIMIZE - used only once
     int cliqOffsetI = subProb_[*cliqI].getCliqOffset(); //OPTIMIZE - used only once
     std::vector<int> memNodeI = subProb_[*cliqI].memNode_; //OPTIMIZE - used only once
     std::vector<int> nodeOffsetI = subProb_[*cliqI].getNodeOffset(); //OPTIMIZE - used only once

     int blkRow = 0;

     if (*cliqI == cliqJ) {

      //std::cout<<"CLIQUE "<<cliqJ<<std::endl;
      for (int elemI = 0; elemI != sizCliqJ; ++elemI) {
       for (int labelI = 0; labelI != nLabelJ[elemI]; ++labelI) {

        double f1 = 0, f2 = 0;

        double nIlISumF1 = nodeLabSum[nodeOffsetI[elemI] + labelI];
        double nJlJSumF1 = nodeLabSum[nodeOffsetI[elemJ] + labelJ];
        double nInJlIlJSumF1;

        if ((elemI == elemJ) && (labelI == labelJ)) {
         f1 = nIlISumF1*(1 -nIlISumF1);
        }
        else if ((elemI == elemJ) && (labelI != labelJ)) {
         f1 = -1*nIlISumF1*nJlJSumF1;
        }
        else {
         if (elemI < elemJ) {
          nInJlIlJSumF1 = nodePairLabSum[(nodeOffsetI[elemI] + labelI)*subDualSiz + nodeOffsetI[elemJ] + labelJ];
         }
         else {
          nInJlIlJSumF1 = nodePairLabSum[(nodeOffsetI[elemJ] + labelJ)*subDualSiz + nodeOffsetI[elemI] + labelI];
         }

         f1 = (nInJlIlJSumF1 - nIlISumF1*nJlJSumF1);
        }

        if (curJNode == memNodeI[elemI]) {

         f2 = f2Arr[labelI];
        }

        int curI = cliqOffsetI + nodeOffsetI[elemI] + labelI;

        double curVal = 0;

        if (curI == curJ) {
         curVal = dampLambda_;
        }

        curVal += tau_*f1 + f2;

         if (cntIter == 1) {
          eigenHess_.insert(curI,curJ) = curVal;
         }
         else {
          eigenHess_.coeffRef(curI,curJ) = curVal;
         }

        if ((curI == curJ) && (dampLambda_ < 0.001)) {
         diagReg[blkRow] = 0.001;
        }

        cliqBlk(blkRow,blkCol) = curVal; //*linSysScale_

        ++blkRow;
       } //labelI
      } //elemI
     } //if *cligI == cliqJ
     else {
      for (int elemI = 0; elemI != sizCliqI; ++elemI) {
       if (curJNode == memNodeI[elemI]) {
        for (int labelI = 0; labelI != nLabelJ[elemI]; ++labelI) {

         int curI = cliqOffsetI + nodeOffsetI[elemI] + labelI;

         if (cntIter == 1) {
          eigenHess_.insert(curI,curJ) = f2Arr[labelI];
         }
         else {
          eigenHess_.coeffRef(curI,curJ) = f2Arr[labelI];
         }
        }
       }
      }
     }

    } //for iterating through cliqNeigh

    delete [] f2Arr;

    ++blkCol;
   } // for labelJ
  } //for elemJ

  for (int iDiag = 0; iDiag != subDualSiz; ++iDiag) {
   cliqBlk(iDiag, iDiag) += diagReg[iDiag];
  }

  blkJacob_[cliqJ] = cliqBlk.inverse();
 } //for cliqJ

 if (cntIter == 1) {
  eigenHess_.makeCompressed();
 }

 //std::cout<<"Hessian matrix: max element "<<maxElem<<" min element "<<minElem<<std::endl;

 if (isnan(curEnergy_)) {
  std::cout<<"ENERGY INDEED NAN!"<<std::endl;
  return -1;
 }

 return 0;
}

int dualSys::popGradEnergyFista() {
 double f2DenExpMax;
 std::vector<double> f2LabelSum(numLabTot_);

 curEnergy_ = 0;
 double curEnergyRdx = 0;

 double stgTerm = 0;

//#pragma omp parallel for reduction(+:curEnergyRdx)
 for (int curNode = 0; curNode < nNode_; ++curNode) {
  double expVal;

  int nLabel = nLabel_[curNode];

  std::vector<double> f2DenExp(nLabel);

  double f2Den = 0;

//  #pragma omp parallel for
  for (int i = 0; i != nLabel; ++i) {
   double cliqSumLabel = 0;

//   #pragma omp for reduction(+ : cliqSumLabel)
   for (std::vector<int>::iterator j = cliqPerNode_[curNode].begin(); j != cliqPerNode_[curNode].end(); ++j) {
    int nodeInd = myUtils::findNodeIndex(subProb_[*j].memNode_, curNode);

    std::vector<double> momentum = subProb_[*j].getMomentum();

    cliqSumLabel += momentum[subProb_[*j].getNodeOffset()[nodeInd] + i];
   }
   f2DenExp[i] = uEnergy_[unaryOffset_[curNode] + i] + cliqSumLabel;
  } //for nLabel

  f2DenExpMax = *std::max_element(f2DenExp.begin(), f2DenExp.end());

  for (int i = 0; i != nLabel; ++i) {
   expVal = exp(tau_*(f2DenExp[i] - f2DenExpMax));
   myUtils::checkRangeError(expVal);

   f2LabelSum[unaryOffset_[curNode] + i] = expVal;

   f2Den += expVal;
  } //for nLabel

  for (int i = 0; i != nLabel; ++i) {
   f2LabelSum[unaryOffset_[curNode] + i] /= f2Den;
  }

  curEnergyRdx += (1/tau_)*log(f2Den) + f2DenExpMax;
 } //for curNode

 double f1Den;

 //std::cout<<"clique processed:";
#if 1
//#pragma omp parallel for reduction(+:curEnergyRdx,stgTerm)
 for (int cliqJ = 0; cliqJ < nCliq_; ++cliqJ) {
  //std::cout<<" "<<cliqJ;
  std::cout<<std::flush;

  int nCliqLab = subProb_[cliqJ].nCliqLab_;
  int sizCliqJ = subProb_[cliqJ].sizCliq_;
  std::vector<short> nLabelJ = subProb_[cliqJ].getNodeLabel();

  std::vector<double> momentum = subProb_[cliqJ].getMomentum();
  //std::vector<double> cEnergy = subProb_[cliqJ].getCE();
  //std::vector<double> cEnergy = cEnergy2DVec_[cliqOffset_[cliqJ]];
  std::vector<int> nodeOffsetJ = subProb_[cliqJ].getNodeOffset();
  int cliqOffsetJ = subProb_[cliqJ].getCliqOffset();
  std::vector<int> memNodeJ = subProb_[cliqJ].memNode_;

  int subDualSiz = momentum.size();

  std::vector<double> nodeLabSum(subDualSiz);

  if (!sparseFlag_) {
   double expVal;

   f1Den = 0;

   std::vector<double> f1DenExp(nCliqLab);

   double f1DenExpMax = 0;

   for (int i = 0; i != nCliqLab; ++i) {
    double nodeSum = 0;

    double labPull = nCliqLab;

    for (int j = 0; j != sizCliqJ; ++j) {
     short cliqLabCur;
     labPull /= nLabelJ[j];

     int labPullTwo = ceil((i+1)/labPull);

     if (labPullTwo % nLabelJ[j] == 0) {
      cliqLabCur = nLabelJ[j] - 1;
     }
     else {
      cliqLabCur = (labPullTwo % nLabelJ[j]) - 1;
     }

     nodeSum += momentum[nodeOffsetJ[j] + cliqLabCur];
    }

    f1DenExp[i] = subProb_[cliqJ].getCE(i) - nodeSum;

    if (i == 0) {
     f1DenExpMax = f1DenExp[i];
    }
    else {
     if (f1DenExp[i] > f1DenExpMax) {
      f1DenExpMax = f1DenExp[i];
     }
    }
   }

 //  double f1DenExpMax = *std::max_element(f1DenExp.begin(), f1DenExp.end());

   for (int i = 0; i != nCliqLab; ++i) {
    expVal = exp(tau_*(f1DenExp[i] - f1DenExpMax));
    myUtils::checkRangeError(expVal);
    f1Den += expVal;

 //   expVal = exp(tau_*(cEnergy[i] - f1NodeSum[i] - f1DenExpMax));
 //   myUtils::checkRangeError(expVal);

    std::vector<short> cliqLab;

    double labPull = nCliqLab;

    for (int j = 0; j != sizCliqJ; ++j) {
     labPull /= nLabelJ[j];

     int labPullTwo = ceil((i+1)/labPull);

     if (labPullTwo % nLabelJ[j] == 0) {
      cliqLab.push_back(nLabelJ[j] - 1);
     }
     else {
      cliqLab.push_back((labPullTwo % nLabelJ[j]) - 1);
     }

     nodeLabSum[nodeOffsetJ[j] + cliqLab[j]] += expVal;
    }
   }

   for (std::vector<double>::iterator nodeLabIter = nodeLabSum.begin(); nodeLabIter != nodeLabSum.end(); ++nodeLabIter) {
    *nodeLabIter /= f1Den;
   }

   curEnergyRdx += (1/tau_)*log(f1Den) + f1DenExpMax;
  }
  else {
   double subEnergy;
   cliqSPSparseFista(&subProb_[cliqJ], tau_, subEnergy, nodeLabSum);
   curEnergyRdx += subEnergy;
  }

  for (int elemJ = 0; elemJ != sizCliqJ; ++elemJ) {
   for (int labelJ = 0; labelJ != nLabelJ[elemJ]; ++labelJ) {
    int curJ = cliqOffsetJ + nodeOffsetJ[elemJ] + labelJ;

    int curJNode = memNodeJ[elemJ];

    gradient_[curJ] = -1*nodeLabSum[nodeOffsetJ[elemJ] + labelJ] + f2LabelSum[unaryOffset_[curJNode] + labelJ];

    if (stgCvxFlag_) {
     gradient_[curJ] += stgCvxCoeff_*dualVar_[curJ];

     stgTerm += pow(dualVar_[curJ],2);
    }

   } // for labelJ
  } //for elemJ
 } //for cliqJ
#endif

 if (stgCvxFlag_) {
  curEnergyRdx += 0.5*stgCvxCoeff_*stgTerm;
  //std::cout<<"popGradEnergy: strong convexity term added to the energy: "<<stgTerm<<std::endl;
 }

 curEnergy_ = curEnergyRdx;

 //std::cout<<std::endl;

 if (isnan(curEnergy_)) {
  std::cout<<"ENERGY INDEED NAN!"<<std::endl;
  return -1;
 }

 return 0;
}

int dualSys::popGradEnergy() {
 double f2DenExpMax;
 std::vector<double> f2LabelSum(numLabTot_);

 curEnergy_ = 0;
 double curEnergyRdx = 0;

 double stgTerm = 0;

 //#pragma omp parallel for reduction(+ : curEnergyRdx)
 for (int curNode = 0; curNode < nNode_; ++curNode) {
  double expVal;

  int nLabel = nLabel_[curNode];

  std::vector<double> f2DenExp(nLabel);

  double f2Den = 0;

//  #pragma omp parallel for
  for (int i = 0; i != nLabel; ++i) {
   double cliqSumLabel = 0;

//   #pragma omp for reduction(+ : cliqSumLabel)
   for (std::vector<int>::iterator j = cliqPerNode_[curNode].begin(); j != cliqPerNode_[curNode].end(); ++j) {
    int nodeInd = myUtils::findNodeIndex(subProb_[*j].memNode_, curNode);

    std::vector<double> dualVar = subProb_[*j].getDualVar();

    cliqSumLabel += dualVar[subProb_[*j].getNodeOffset()[nodeInd] + i];
   }
   f2DenExp[i] = uEnergy_[unaryOffset_[curNode] + i] + cliqSumLabel;
  } //for nLabel

  f2DenExpMax = *std::max_element(f2DenExp.begin(), f2DenExp.end());

  for (int i = 0; i != nLabel; ++i) {
   expVal = exp(tau_*(f2DenExp[i] - f2DenExpMax));
   myUtils::checkRangeError(expVal);

   f2LabelSum[unaryOffset_[curNode] + i] = expVal;

   f2Den += expVal;
  } //for nLabel

  for (int i = 0; i != nLabel; ++i) {
   f2LabelSum[unaryOffset_[curNode] + i] /= f2Den;
  }

  curEnergyRdx += (1/tau_)*log(f2Den) + f2DenExpMax;
 } //for curNode

 double f1Den;

#if 1
 //#pragma omp parallel for reduction(+ : curEnergyRdx,stgTerm)
 for (int cliqJ = 0; cliqJ < nCliq_; ++cliqJ) {
  int nCliqLab = subProb_[cliqJ].nCliqLab_;
  int sizCliqJ = subProb_[cliqJ].sizCliq_;
  std::vector<short> nLabelJ = subProb_[cliqJ].getNodeLabel();

  std::vector<double> dualVar = subProb_[cliqJ].getDualVar();
  //std::vector<double> cEnergy = subProb_[cliqJ].getCE();
  //std::vector<double> cEnergy = cEnergy2DVec_[cliqOffset_[cliqJ]];
  std::vector<int> nodeOffsetJ = subProb_[cliqJ].getNodeOffset();
  int cliqOffsetJ = subProb_[cliqJ].getCliqOffset();
  std::vector<int> memNodeJ = subProb_[cliqJ].memNode_;

  int subDualSiz = dualVar.size();

  std::vector<double> nodeLabSum(subDualSiz);

  if (!sparseFlag_) {
   double expVal;

   f1Den = 0;

   std::vector<double> f1DenExp(nCliqLab);

   double f1DenExpMax = 0;

   for (int i = 0; i != nCliqLab; ++i) {
    double nodeSum = 0;

    double labPull = nCliqLab;

    for (int j = 0; j != sizCliqJ; ++j) {
     short cliqLabCur;
     labPull /= nLabelJ[j];

     int labPullTwo = ceil((i+1)/labPull);

     if (labPullTwo % nLabelJ[j] == 0) {
      cliqLabCur = nLabelJ[j] - 1;
     }
     else {
      cliqLabCur = (labPullTwo % nLabelJ[j]) - 1;
     }

     nodeSum += dualVar[nodeOffsetJ[j] + cliqLabCur];
    }

    f1DenExp[i] = subProb_[cliqJ].getCE(i) - nodeSum;

    if (i == 0) {
     f1DenExpMax = f1DenExp[i];
    }
    else {
     if (f1DenExp[i] > f1DenExpMax) {
      f1DenExpMax = f1DenExp[i];
     }
    }
   }

 //  double f1DenExpMax = *std::max_element(f1DenExp.begin(), f1DenExp.end());

   for (int i = 0; i != nCliqLab; ++i) {
    expVal = exp(tau_*(f1DenExp[i] - f1DenExpMax));
    myUtils::checkRangeError(expVal);
    f1Den += expVal;

 //   expVal = exp(tau_*(cEnergy[i] - f1NodeSum[i] - f1DenExpMax));
 //   myUtils::checkRangeError(expVal);

    std::vector<short> cliqLab;

    double labPull = nCliqLab;

    for (int j = 0; j != sizCliqJ; ++j) {
     labPull /= nLabelJ[j];

     int labPullTwo = ceil((i+1)/labPull);

     if (labPullTwo % nLabelJ[j] == 0) {
      cliqLab.push_back(nLabelJ[j] - 1);
     }
     else {
      cliqLab.push_back((labPullTwo % nLabelJ[j]) - 1);
     }

     nodeLabSum[nodeOffsetJ[j] + cliqLab[j]] += expVal;
    }
   }

   for (std::vector<double>::iterator nodeLabIter = nodeLabSum.begin(); nodeLabIter != nodeLabSum.end(); ++nodeLabIter) {
    *nodeLabIter /= f1Den;
   }

   curEnergyRdx += (1/tau_)*log(f1Den) + f1DenExpMax;
  }
  else {
   double subEnergy;
   cliqSPSparseGradEnergy(&subProb_[cliqJ], dualVar, tau_, subEnergy, nodeLabSum);
   curEnergyRdx += subEnergy;

   if (isnan(curEnergy_)) {
    std::cout<<"ENERGY INDEED NAN!"<<std::endl;
    return -1;
   }
  }

  for (int elemJ = 0; elemJ != sizCliqJ; ++elemJ) {
   for (int labelJ = 0; labelJ != nLabelJ[elemJ]; ++labelJ) {
    int curJ = cliqOffsetJ + nodeOffsetJ[elemJ] + labelJ;

    int curJNode = memNodeJ[elemJ];

    gradient_[curJ] = -1*nodeLabSum[nodeOffsetJ[elemJ] + labelJ] + f2LabelSum[unaryOffset_[curJNode] + labelJ];

    if (stgCvxFlag_) {
     gradient_[curJ] += stgCvxCoeff_*dualVar_[curJ];

     stgTerm += pow(dualVar_[curJ],2);
    }

   } // for labelJ
  } //for elemJ
 } //for cliqJ
#endif

 if (stgCvxFlag_) {
  curEnergyRdx += 0.5*stgCvxCoeff_*stgTerm;
  //std::cout<<"popGradEnergy: strong convexity term added to the energy: "<<stgTerm<<std::endl;
 }

 curEnergy_ = curEnergyRdx;

 if (isnan(curEnergy_)) {
  std::cout<<"ENERGY INDEED NAN!"<<std::endl;
  return -1;
 }

 return 0;
}

#if 1
int dualSys::popGradEnergy(const std::vector<double> &ipVar, std::vector<double> &opGrad, double &opFunc, double &opGradNorm, double &opGradMax) {
 opGrad.resize(nDualVar_,0);

 std::vector<std::vector<double> > ipVarDist(nCliq_);

 for (int i = 0; i != nCliq_; ++i) {
  std::copy(ipVar.begin() + subProb_[i].getCliqOffset(), ipVar.begin() + subProb_[i].getCliqOffset() + subProb_[i].getDualSiz(), std::back_inserter(ipVarDist[i]));
 }

 double f2DenExpMax;
 std::vector<double> f2LabelSum(numLabTot_);

 opFunc = 0;

 double stgTerm = 0;

 //#pragma omp parallel for reduction(+ : opFunc)
 for (int curNode = 0; curNode < nNode_; ++curNode) {
  double expVal;

  int nLabel = nLabel_[curNode];

  std::vector<double> f2DenExp(nLabel);

  double f2Den = 0;

//  #pragma omp parallel for
  for (int i = 0; i != nLabel; ++i) {
   double cliqSumLabel = 0;

//   #pragma omp for reduction(+ : cliqSumLabel)
   for (std::vector<int>::iterator j = cliqPerNode_[curNode].begin(); j != cliqPerNode_[curNode].end(); ++j) {
    int nodeInd = myUtils::findNodeIndex(subProb_[*j].memNode_, curNode);

    cliqSumLabel += ipVarDist[*j][subProb_[*j].getNodeOffset()[nodeInd] + i];
   }
   f2DenExp[i] = uEnergy_[unaryOffset_[curNode] + i] + cliqSumLabel;
  } //for nLabel

  f2DenExpMax = *std::max_element(f2DenExp.begin(), f2DenExp.end());

  for (int i = 0; i != nLabel; ++i) {
   expVal = exp(tau_*(f2DenExp[i] - f2DenExpMax));
   myUtils::checkRangeError(expVal);

   f2LabelSum[unaryOffset_[curNode] + i] = expVal;

   f2Den += expVal;
  } //for nLabel

  for (int i = 0; i != nLabel; ++i) {
   f2LabelSum[unaryOffset_[curNode] + i] /= f2Den;
  }

  opFunc += (1/tau_)*log(f2Den) + f2DenExpMax;
 } //for curNode

#if 1
 //#pragma omp parallel for reduction(+ : opFunc)
 for (int cliqJ = 0; cliqJ < nCliq_; ++cliqJ) {
  int sizCliqJ = subProb_[cliqJ].sizCliq_;
  std::vector<short> nLabelJ = subProb_[cliqJ].getNodeLabel();

  //std::vector<double> cEnergy = subProb_[cliqJ].getCE();
  //std::vector<double> cEnergy = cEnergy2DVec_[cliqOffset_[cliqJ]];
  std::vector<int> nodeOffsetJ = subProb_[cliqJ].getNodeOffset();
  int cliqOffsetJ = subProb_[cliqJ].getCliqOffset();
  std::vector<int> memNodeJ = subProb_[cliqJ].memNode_;

  int subDualSiz = subProb_[cliqJ].getDualSiz();

  std::vector<double> nodeLabSum(subDualSiz);

  double subEnergy;

  cliqSPSparseGradEnergy(&subProb_[cliqJ], ipVarDist[cliqJ], tau_, subEnergy, nodeLabSum);

  opFunc += subEnergy;

  for (int elemJ = 0; elemJ != sizCliqJ; ++elemJ) {
   for (int labelJ = 0; labelJ != nLabelJ[elemJ]; ++labelJ) {
    int curJ = cliqOffsetJ + nodeOffsetJ[elemJ] + labelJ;

    int curJNode = memNodeJ[elemJ];

    opGrad[curJ] = -1*nodeLabSum[nodeOffsetJ[elemJ] + labelJ] + f2LabelSum[unaryOffset_[curJNode] + labelJ];

    if (stgCvxFlag_) {
     gradient_[curJ] += stgCvxCoeff_*dualVar_[curJ];

     stgTerm += pow(dualVar_[curJ],2);
    }
   } // for labelJ
  } //for elemJ
 } //for cliqJ
#endif

 if (stgCvxFlag_) {
  opFunc += 0.5*stgCvxCoeff_*stgTerm;
 }

 opGradNorm = myUtils::norm<double>(opGrad, nDualVar_);

 int gradMaxInd = myUtils::argmaxAbs<double>(opGrad, 0, nDualVar_);

 opGradMax = opGrad[gradMaxInd];

 if (isnan(curEnergy_)) {
  std::cout<<"ENERGY INDEED NAN!"<<std::endl;
  return -1;
 }

 return 0;
}
#endif

int dualSys::performLineSearch() {
 std::vector<double> z(nDualVar_);

 std::vector<double> oriNewtonStep = newtonStep_;

 myUtils::addArray<double>(dualVar_,newtonStep_,z,nDualVar_);

 double iterEnergy = (sparseFlag_) ? compEnergySparse(z):compEnergy(z);

 int lsCnt = 0;

 std::cout<<"performLineSearch: LHS energy "<<iterEnergy<<" RHS energy = "<<curEnergy_<<" + "<<lsC_<<" * "<<myUtils::dotVec(gradient_,newtonStep_,nDualVar_)<<" "<<lsTol_<<std::endl;

 while (iterEnergy > curEnergy_ + lsC_*(myUtils::dotVec<double>(gradient_,newtonStep_,nDualVar_)) + lsTol_) {
  ++lsCnt;

  lsAlpha_ *= lsRho_;

  myUtils::scaleVector<double>(lsAlpha_,oriNewtonStep,newtonStep_,nDualVar_);
  myUtils::addArray<double>(dualVar_,newtonStep_,z,nDualVar_);

  iterEnergy = (sparseFlag_) ? compEnergySparse(z):compEnergy(z);

  double rhsEnergy = curEnergy_ + lsC_*(myUtils::dotVec<double>(gradient_,newtonStep_,nDualVar_)) + lsTol_;
  std::cout<<"performLineSearch: "<<lsCnt<<" lhs "<<iterEnergy<<" rhs "<<rhsEnergy<<std::endl;
 }

 lsAlpha_ = 1;

 return lsCnt;
}

int dualSys::performLineSearch(std::vector<double> &newGrad, double &newFun) {
 lsAlpha_ = 1;

 int lsCnt = 0;

 double gradNorm, gradMax, newtNorm;

 std::vector<double> oriNewtonStep = newtonStep_;
 std::vector<double> unitNewtonStep = newtonStep_;

 newtNorm = myUtils::norm(oriNewtonStep,nDualVar_);

 myUtils::scaleVector<double>((1/newtNorm),oriNewtonStep,unitNewtonStep,nDualVar_);

 //myUtils::scaleVector<double>(lsAlpha_, oriNewtonStep, newtonStep_, nDualVar_);

 std::vector<double> newDual(nDualVar_);

 myUtils::addArray(dualVar_, oriNewtonStep, newDual, nDualVar_);

 popGradEnergy(newDual, newGrad, newFun, gradNorm, gradMax);

 bool contLS = true;

 std::cout<<"performLineSearch::interpolated: lhs "<<newFun<<" RHS energy = "<<curEnergy_<<" + "<<lsC_<<" * "<<(myUtils::dotVec(gradient_, newtonStep_, nDualVar_))<<std::endl;

 std::vector<double> pointOne = {0, curEnergy_, myUtils::dotVec(gradient_, unitNewtonStep, nDualVar_)};

 while ((contLS) && (newFun > curEnergy_ + lsC_*(myUtils::dotVec(gradient_, newtonStep_, nDualVar_)) + lsTol_)) {

  double tempScale = lsAlpha_;

  std::vector<double> pointTwo = {lsAlpha_, newFun, myUtils::dotVec(newGrad, unitNewtonStep, nDualVar_)};

  std::vector<std::vector<double> > points = {pointOne, pointTwo};

  lsAlpha_ = polyInterp(points);

  if (lsAlpha_ < tempScale*pow(10,-3)) {
   lsAlpha_ = tempScale*pow(10,-3);
  }
  else if (lsAlpha_ > tempScale*0.6) {
   lsAlpha_ = tempScale*0.6;
  }

  myUtils::scaleVector<double>(lsAlpha_, oriNewtonStep, newtonStep_, nDualVar_);

  double l1Norm = myUtils::l1Norm(newtonStep_);

  if (l1Norm < lsTol_) {
   contLS = false;
   newFun = curEnergy_;
   newGrad = gradient_;
   lsAlpha_ = 0;
  }

  myUtils::scaleVector<double>(lsAlpha_, oriNewtonStep, newtonStep_, nDualVar_);

  myUtils::addArray(dualVar_, newtonStep_, newDual, nDualVar_);

  popGradEnergy(newDual, newGrad, newFun, gradNorm, gradMax);

  std::cout<<"performLineSearch::interpolated: lhs "<<newFun<<" RHS energy = "<<curEnergy_<<" + "<<lsC_<<" * "<<(myUtils::dotVec(gradient_, newtonStep_, nDualVar_))<<" lsAlpha "<<lsAlpha_<<std::endl;

  ++lsCnt;
 }

 return lsCnt;
}

int dualSys::solveNewton()
{
 std::cout<<"solveNewton: Sparse flag "<<sparseFlag_<<std::endl;
 std::cout<<"solveNewton: Full Hessian store flag "<<fullHessStoreFlag_<<std::endl;

 precondFlag_ = 1; //0: Jacobi; 1: Block Jacobi; 2: Incomplete Cholesky; 3: Quasi-Newton.

 if (precondFlag_ == 2) { //create Incomplete Cholesky data-structures
  inCholDiag_ = new double[nDualVar_];
  inCholColPtr_ = new int[nDualVar_ + 1];
 }
 else if (precondFlag_ == 3) { //create quasi Newton data-structures
  sVec_ = new double[nDualVar_]();
  yVec_ = new double[nDualVar_]();
  resVec_ = new double[nDualVar_];
  prVec_ = new double[nDualVar_];

  numVarQN_ = nDualVar_;
  numPairQN_ = 20;
  iopQN_ = 1;
  workVecQN_ = new double[4*(numPairQN_+1)*numVarQN_ + 2*(numPairQN_+1) + 1](); //4(M+1)N + 2(M+1) + 1
  lwQN_ = 4*(numPairQN_+1)*numVarQN_ + 2*(numPairQN_+1) + 1;
  iWorkVecQN_ = new int[2*numPairQN_ + 30](); //2M + 30
  liwQN_ = 2*numPairQN_ + 30;
  buildQN_ = true;
 }

 double tCompEnergies = 0;

 bool contIter = true;

 int cntIter = 0, cntIterTau = 0;

 double gradBasedDamp = 0;

 dampLambda_ = dampLambdaInit_; //####

 Eigen::VectorXd eigenGuess = Eigen::VectorXd::Zero(nDualVar_);

 double bestPrimalEnergy = 0;

 if (fullHessStoreFlag_) {
  eigenHess_.reserve(reserveHess_);
 }
 else {
  cliqBlkHess_.reserve(reserveCliqBlk_);
 }

 timeStart_ = myUtils::getTime();

 while ((contIter) && (cntIter < maxIter_)) {
  double tFull = myUtils::getTime();

  ++cntIter;
  ++cntIterTau;

  double tGHE = myUtils::getTime(); //time to comp gradient, hessian and current energy

  if (fullHessStoreFlag_) {
   popGradHessEnergy(cntIter);
  }
  else {
   popGradHessEnergyPerf(cntIter);
  }

  double gradNorm = myUtils::norm<double>(gradient_, nDualVar_);

  if (cntIter == 1) {
   gradBasedDamp = gradNorm/gradDampFactor_; //earlier, it was 3 ####
  }

  if ((annealIval_ != -1) && (gradNorm < gradBasedDamp) && (tau_ < tauMax_)) {
   std::cout<<"Annealing: grad norm "<<gradNorm<<" grad based damp "<<gradBasedDamp<<std::endl;
   cntIterTau = 1;
   tau_ *= tauScale_;
//   gradDampFlag = true;
//   if (tau_ == tauMax_) {
//    dampLambda_ = 10*dampLambdaInit_;
//   }
//   else {
   dampLambda_ = dampLambdaInit_; //####
//   }
//   eigenHess_.setZero();
//   eigenHess_.reserve(reserveHess_);

   if (fullHessStoreFlag_) {
    popGradHessEnergy(cntIter);
   }
   else {
    popGradHessEnergyPerf(cntIter);
   }

   gradNorm = myUtils::norm<double>(gradient_, nDualVar_);
   gradBasedDamp = gradNorm/gradDampFactor_;

   if (gradBasedDamp < dampTol_) {
    gradBasedDamp = dampTol_;
   }
  }

  int gradMaxInd = myUtils::argmaxAbs<double>(gradient_,0,nDualVar_);

  double gradMax = std::abs(gradient_[gradMaxInd]);

  std::cout<<"ITERATION "<<cntIter<<" starts, with tau "<<tau_<<" and Gradient threshold "<<gradBasedDamp<<"."<<std::endl;
  std::cout<<"solveNewton: populating gradient, hessian and energy took "<<(myUtils::getTime() - tGHE)<<" seconds."<<std::endl;
  std::cout<<"solveNewton: gradient l-infinity norm: "<<gradMax<<", Euclidean norm: "<<gradNorm<<". Energy: "<<curEnergy_<<std::endl;

  int checkIter = -1;

  std::string opName;

  if (cntIter == checkIter) {
   opName = "gradient_" + std::to_string(checkIter) + ".txt";

   std::ofstream opGrad(opName);
   opGrad<<std::scientific;
   opGrad<<std::setprecision(6);
   for (std::vector<double>::iterator gradIter = gradient_.begin(); gradIter != gradient_.end(); ++gradIter) {
    opGrad<<*gradIter<<std::endl;
   }
   opGrad.close();

   opName = "dual_" + std::to_string(checkIter) + ".txt";

   std::ofstream opDual(opName);
   opDual<<std::scientific;
   opDual<<std::setprecision(6);
   for (std::vector<double>::iterator dualIter = dualVar_.begin(); dualIter != dualVar_.end(); ++dualIter) {
    opDual<<*dualIter<<std::endl;
   }
   opDual.close();

#if 1
   std::ofstream hessFile;

   opName = "hessian_" + std::to_string(checkIter) + ".txt";

   hessFile.open(opName);
   hessFile<<std::scientific;
   hessFile<<std::setprecision(6);

   //std::ofstream fullHessian("trueHessian.txt");

   //for (int iRow = 0; iRow != nDualVar_; ++iRow) {
   // for (int iCol = 0; iCol != (nDualVar_-1); ++iCol) {
   //  if (iRow < iCol) {
   //   fullHessian<<eigenHess_.coeffRef(iCol,iRow)<<" ";
   //  }
   //  else {
   //   fullHessian<<eigenHess_.coeffRef(iRow,iCol)<<" ";
   //  }
   // }

   // fullHessian<<eigenHess_.coeffRef(nDualVar_-1,iRow)<<std::endl;
   //}

   //fullHessian.close();

   for (int cJ = 0; cJ != nCliq_; ++cJ) {
    int sizCliqJ = subProb_[cJ].sizCliq_;
    int offsetOne = subProb_[cJ].getCliqOffset();
    std::set<int> cliqNeigh = subProb_[cJ].cliqNeigh_;

    std::vector<int> nodeOffsetJ = subProb_[cJ].getNodeOffset();
    std::vector<short> nLabelJ = subProb_[cJ].getNodeLabel();

    for (int nJ = 0; nJ != sizCliqJ; ++nJ) {
     for (int lJ = 0; lJ != nLabelJ[nJ]; ++lJ) {
      int curJ = offsetOne + nodeOffsetJ[nJ] + lJ;

      for (std::set<int>::iterator cI = cliqNeigh.begin(); cI != cliqNeigh.end(); ++cI) {
       std::vector<int> nodeOffsetI = subProb_[*cI].getNodeOffset();
       std::vector<short> nLabelI = subProb_[*cI].getNodeLabel();

       for (int nI = 0; nI != subProb_[*cI].sizCliq_; ++nI) {
        for (int lI = 0; lI != nLabelI[nI]; ++lI) {
         int curI = subProb_[*cI].getCliqOffset() + nodeOffsetI[nI] + lI;

         hessFile<<curI<<" "<<curJ<<" "<<eigenHess_.coeffRef(curI,curJ)<<std::endl;
         if (eigenHess_.coeffRef(curI,curJ) != 0) {
          //std::cout<<curI<<" "<<curJ<<" "<<eigenHess_.coeffRef(curI,curJ)<<std::endl;
         }
        }
       }
      }
     }
    }
   }

   hessFile.close();

#endif

#if 0
   std::cout<<"Memory Optimized Hessian"<<std::endl;
   std::cout.precision(16);
   std::cout<<std::scientific;

   int probCnt1 = 0, probCnt2 = 0, probCnt3 = 0, probCnt4 = 0, probCnt5 = 0;

   for (int cJ = 0; cJ != nCliq_; ++cJ) {
    int sizCliqJ = subProb_[cJ].sizCliq_;
    int offsetOne = subProb_[cJ].getCliqOffset();
    std::set<int> cliqNeigh = subProb_[cJ].cliqNeigh_;

    std::vector<int> nodeOffsetJ = subProb_[cJ].getNodeOffset();
    std::vector<short> nLabelJ = subProb_[cJ].getNodeLabel();
    std::vector<int> memNodeJ = subProb_[cJ].getMemNode();

    for (int nJ = 0; nJ != sizCliqJ; ++nJ) {
     for (int lJ = 0; lJ != nLabelJ[nJ]; ++lJ) {
      int curJ = offsetOne + nodeOffsetJ[nJ] + lJ;

      for (std::set<int>::iterator cI = cliqNeigh.begin(); cI != cliqNeigh.end(); ++cI) {
       std::vector<int> nodeOffsetI = subProb_[*cI].getNodeOffset();
       std::vector<short> nLabelI = subProb_[*cI].getNodeLabel();
       std::vector<int> memNodeI = subProb_[*cI].getMemNode();

       for (int nI = 0; nI != subProb_[*cI].sizCliq_; ++nI) {
        for (int lI = 0; lI != nLabelI[nI]; ++lI) {
         int curI = subProb_[*cI].getCliqOffset() + nodeOffsetI[nI] + lI;

         //hessFile<<curI<<" "<<curJ<<" "<<eigenHess_.coeffRef(curI,curJ)<<std::endl;
         if ((*cI == cJ) && (eigenHess_.coeffRef(curI,curJ) != 0)) {
          if ((memNodeI[nI] == memNodeJ[nJ])) {
           if (lI >= lJ) {
            //std::cout<<curI<<" "<<curJ<<" "<<cliqBlkHess_.coeffRef(curI,curJ) + nodeBlkHess_[memNodeI[nI]](lI,lJ)<<std::endl;
            //if (eigenHess_.coeffRef(curI,curJ) != (cliqBlkHess_.coeffRef(curI,curJ) + nodeBlkHess_[memNodeI[nI]](lI,lJ))) {
            if (eigenHess_.coeffRef(curI,curJ) != blkDiag_[cJ](nodeOffsetI[nI] + lI,nodeOffsetJ[nJ] + lJ)) {
             //std::cout<<"PROBLEM 1! "<<eigenHess_.coeffRef(curI,curJ)<<" "<<(cliqBlkHess_.coeffRef(curI,curJ) + nodeBlkHess_[memNodeI[nI]](lI,lJ))<<std::endl;
             double tempVal1 = eigenHess_.coeffRef(curI,curJ);
             double tempVal2 = blkDiag_[cJ](nodeOffsetI[nI] + lI,nodeOffsetJ[nJ] + lJ);
             ++probCnt1;
            }
           }
           else {
            //std::cout<<curI<<" "<<curJ<<" "<<cliqBlkHess_.coeffRef(curI,curJ) + nodeBlkHess_[memNodeI[nI]](lJ,lI)<<std::endl;
            //if (eigenHess_.coeffRef(curI,curJ) != (cliqBlkHess_.coeffRef(curI,curJ) + nodeBlkHess_[memNodeI[nI]](lJ,lI))) {
            if (eigenHess_.coeffRef(curI,curJ) != blkDiag_[cJ](nodeOffsetI[nI] + lI,nodeOffsetJ[nJ] + lJ)) {
             //std::cout<<"PROBLEM 2! "<<eigenHess_.coeffRef(curI,curJ)<<" "<<(cliqBlkHess_.coeffRef(curI,curJ) + nodeBlkHess_[memNodeI[nI]](lJ,lI))<<std::endl;
             ++probCnt2;
            }
           }
          }
          else {
           //std::cout<<curI<<" "<<curJ<<" "<<cliqBlkHess_.coeffRef(curI,curJ)<<std::endl;
           if (eigenHess_.coeffRef(curI,curJ) != blkDiag_[cJ](nodeOffsetI[nI] + lI,nodeOffsetJ[nJ] + lJ)) {
            //std::cout<<"PROBLEM 3! "<<eigenHess_.coeffRef(curI,curJ)<<" "<<cliqBlkHess_.coeffRef(curI,curJ)<<std::endl;
            ++probCnt3;
           }
          }
         }
         else if ((*cI != cJ) && (memNodeI[nI] == memNodeJ[nJ])) {
          if (lI >= lJ) {
           //std::cout<<curI<<" "<<curJ<<" "<<nodeBlkHess_[memNodeI[nI]](lI,lJ)<<std::endl;
           if (eigenHess_.coeffRef(curI,curJ) != nodeBlkHess_[memNodeI[nI]](lI,lJ)) {
            //std::cout<<"PROBLEM 4! "<<eigenHess_.coeffRef(curI,curJ)<<" "<<nodeBlkHess_[memNodeI[nI]](lI,lJ)<<std::endl;
            ++probCnt4;
           }
          }
          else {
           //std::cout<<curI<<" "<<curJ<<" "<<nodeBlkHess_[memNodeI[nI]](lJ,lI)<<std::endl;
           if (eigenHess_.coeffRef(curI,curJ) != nodeBlkHess_[memNodeI[nI]](lJ,lI)) {
            //std::cout<<"PROBLEM 5! "<<eigenHess_.coeffRef(curI,curJ)<<" "<<nodeBlkHess_[memNodeI[nI]](lJ,lI)<<std::endl;
            ++probCnt5;
           }
          }
         }
        }
       }
      }
     }
    }
   }
#endif
  }

  if ((gradMax > gradTol_) && (gradNorm > gradTol_) && (pdGap_ > gradTol_) && (smallGapIterCnt_ < 10*cntExitCond_)) {
   Eigen::VectorXd eigenStep(nDualVar_);

   Eigen::VectorXd oriGrad(nDualVar_);

   //custom CG implementation
   for (std::size_t i = 0; i < nDualVar_; ++i) {
    oriGrad[i] = gradient_[i];
    eigenGrad_[i] = gradient_[i];
   }

   double tCG = myUtils::getTime();
   std::vector<Eigen::VectorXd> stepBackVec; //for back tracking
   eigenStep = eigenGuess;
   solveCG(eigenStep, stepBackVec, cntIter);

   std::cout<<"CG took "<<myUtils::getTime() - tCG<<" seconds."<<std::endl;
   std::cout<<std::flush;

   if (precondFlag_ == 2) { //Incomplete Cholesky Preconditioner data structures have to be freed
    delete[] inCholLow_;
    delete[] inCholDiag_;
    delete[] inCholColPtr_;
    delete[] inCholRowInd_;
   }
   else if (precondFlag_ == 3) { //Quasi-Newton preconditioner data structures have to be freed
    if (cntIterTau % 10 == 0) {
     delete[] sVec_;
     delete[] yVec_;
     delete[] resVec_;
     delete[] prVec_;
    }
   }

   for (std::size_t i = 0; i != nDualVar_; ++i) {
    newtonStep_[i] = eigenStep[i];
    eigenGuess[i] = 0; //eigenStep[i]; //####
   }

   Eigen::VectorXd eigenInter = getHessVecProd(eigenStep);

   double interValOne = eigenStep.dot(eigenInter);

   interValOne *= 0.5;

   double interValTwo = eigenStep.dot(oriGrad);

   double nxtApproxEnergyDiff = interValOne + interValTwo; //diff. wrt current energy

   std::vector<double> z(nDualVar_);

   myUtils::addArray<double>(dualVar_,newtonStep_,z,nDualVar_);

   double nxtEnergy = (sparseFlag_) ? compEnergySparse(z):compEnergy(z);

   double rho = (nxtEnergy - curEnergy_)/nxtApproxEnergyDiff;

   if ((isnan(rho)) || (isinf(rho))) {
    std::cout<<"rho debug: nxtApproxEnergyDiff "<<nxtApproxEnergyDiff<<" nxtEnergy "<<nxtEnergy<<" curEnergy_ "<<curEnergy_<<std::endl;
   }

   double eta_1 = 0.25, eta_2 = 0.5, eta_3 = 0.9, sigma_1 = 2, sigma_2 = 0.5, sigma_3 = 0.25, eta_0 = 0.0001;

//updating damping lambda inspired by "Newton's method for large-scale optimization" by Bouaricha et al

//std::cout<<"next energy "<<nxtEnergy<<" "<<" current energy "<<curEnergy_<<" next approx energy diff "<<nxtApproxEnergyDiff<<" interValOne "<<interValOne<<" interValTwo "<<interValTwo<<std::endl;

//  int newtonMaxInd = myUtils::argmaxAbs<double>(newtonStep_,0,nDualVar_);
//  std::cout<<"solveNewton: before line search newton step l-infinity norm: "<<newtonStep_[newtonMaxInd]<<". Euclidean norm: "<<newtonStepNorm<<std::endl;
   double stepNormOld = 0, stepNormNew = 0;

   double tLS = myUtils::getTime(); //time to perform line search

   int lsCnt = -1;

   if (rho <= eta_0) {
    stepNormOld = myUtils::norm<double>(newtonStep_, nDualVar_);

    std::vector<double> oriNewton = newtonStep_;
//    lsCnt = performCGBacktrack(stepBackVec);
//    std::vector<double> resetStep = newtonStep_;
//    newtonStep_ = oriNewton;

//    if (tau_ == tauMax_) {
//     lsCnt = performLineSearch(); //perform line search only when rho <= eta_0
     //newtonStep_ = oriNewton;
//    }
//    else {
     //interpolation based line search
     std::vector<double> newGrad;
     double newFun;
     lsCnt = performLineSearch(newGrad, newFun);
     //myUtils::scaleVector<double>(lsAlpha_,newtonStep_,newtonStep_,nDualVar_);
//    }

    stepNormNew = myUtils::norm<double>(newtonStep_, nDualVar_);

    if (stepNormOld/stepNormNew > 8) {
     dampLambda_ *= pow(sigma_1,2);
    }
    else if (stepNormOld/stepNormNew > 4) {
     dampLambda_ *= pow(sigma_1,2);
    }
    else {
     dampLambda_ *= sigma_1;
    }
   }
   else if (rho <= eta_1) {
    dampLambda_ *= sigma_1;
   }
   else if ((rho > eta_1) && (rho <= eta_2)) {
    dampLambda_ = dampLambda_;
   }
   else if ((rho > eta_2) && (rho <= eta_3)) {
    dampLambda_ *= sigma_2;
   }
   else if (rho > eta_3) {
    dampLambda_ *= sigma_3;
   }

   std::cout<<"solveNewton:trust region param rho "<<rho<<" updated lambda "<<dampLambda_<<std::endl;
//   int newtonMaxInd = myUtils::argmaxAbs<double>(newtonStep_,0,nDualVar_);
//   std::cout<<"solveNewton: after line search newton step l-infinity norm: "<<newtonStep_[newtonMaxInd]<<". Euclidean norm: "<<newtonStepNorm<<std::endl;
   std::cout<<"solveNewton: performed line search "<<lsCnt<<" times, "<<(myUtils::getTime() - tLS)<<" seconds."<<std::endl;
   myUtils::addArray<double>(dualVar_,newtonStep_,dualVar_,nDualVar_);
//   int dualVarInd = myUtils::argmaxAbs<double>(dualVar_,0,nDualVar_);
//   double dualNorm = myUtils::norm<double>(dualVar_, nDualVar_);
//   std::cout<<"solveNewton: dual l-infinity norm: "<<dualVar_[dualVarInd]<<". Euclidean norm: "<<dualNorm<<std::endl;
   distributeDualVars();
   //eigenHess_.setZero(); //change logic to insert in first iteration and coeffRef in subsequent ones.
  }
  else if ((annealIval_ == -1) || (tau_ == tauMax_)) {
   contIter = false;

   gradNorm = myUtils::norm<double>(gradient_, nDualVar_);
   gradMaxInd = myUtils::argmaxAbs<double>(gradient_,0,nDualVar_);
   gradMax = std::abs(gradient_[gradMaxInd]);
//   std::cout<<"solveNewton: gradient l-infinity norm: "<<gradMax<<", Euclidean norm: "<<gradNorm<<". Energy: "<<curEnergy_<<std::endl;
  }
  else {
   gradBasedDamp = 2*gradNorm;

   if (gradBasedDamp < 0.000001) {
    gradBasedDamp = 0.000001;
   }

   //eigenHess_.setZero();
  }

  int cntInterval = 1;

  if ((gradMax > gradTol_) && (((annealIval_ == -1) || (tau_ == tauMax_)) && (cntIter % cntExitCond_ == 0))) {
   recoverFracPrimal();
   recoverFeasPrimal();

   double curNonSmoothPrimalEnergy = compNonSmoothPrimalEnergy();
   double curNonSmoothDualEnergy = compNonSmoothDualEnergy();

   if (pdInitFlag_) {
    bestNonSmoothPrimalEnergy_ = curNonSmoothPrimalEnergy;
    pdInitFlag_ = false;
   }
   else if (curNonSmoothPrimalEnergy > bestNonSmoothPrimalEnergy_) {
    bestNonSmoothPrimalEnergy_ = curNonSmoothPrimalEnergy;
   }

   pdGap_ = (curNonSmoothDualEnergy - bestNonSmoothPrimalEnergy_)/(std::abs(curNonSmoothDualEnergy) + std::abs(bestNonSmoothPrimalEnergy_));

   if (pdGap_ < 10*gradTol_) {
    ++smallGapIterCnt_;
   }

   std::cout<<"solveNewton: curNonSmoothDualEnergy "<<curNonSmoothDualEnergy<<" curNonSmoothPrimalEnergy "<<curNonSmoothPrimalEnergy<<" bestNonSmoothPrimalEnergy "<<bestNonSmoothPrimalEnergy_<<std::endl;
   std::cout<<"solveNewton: PD Gap "<<pdGap_<<std::endl;
  }

#if 1
  if (cntIter % cntInterval == 0) {
   double tEnergy = myUtils::getTime();

   //recoverFracPrimal();
   //recoverFeasPrimal();
   //setFracAsFeas();

   std::cout<<std::fixed;
   std::cout<<std::setprecision(6);

   //std::cout<<"solveNewton: Smooth primal energy "<<compSmoothPrimalEnergy()<<" Smooth dual energy "<<curEnergy_;
   std::cout<<"solveNewton: Smooth dual energy "<<curEnergy_;

   recoverNodeFracPrimal();
   recoverMaxPrimal(primalFrac_);

   double curIntPrimalEnergy;

   curIntPrimalEnergy = compIntPrimalEnergy();

   std::cout<<" Current integral primal energy "<<curIntPrimalEnergy;

   if (bestPrimalEnergy < curIntPrimalEnergy) {
    bestPrimalEnergy = curIntPrimalEnergy;
    bestPrimalMax_ = primalMax_;
    timeToBestPrimal_ = myUtils::getTime() - timeStart_;
    std::cout<<" Best primal updated in iteration "<<cntIter<<" at time "<<timeToBestPrimal_<<std::endl;
   }
   else if (cntIter == cntInterval) {
    bestPrimalEnergy = curIntPrimalEnergy;
    bestPrimalMax_ = primalMax_;
    timeToBestPrimal_ = myUtils::getTime() - timeStart_;
   }

   std::cout<<" Best integral primal energy "<<bestPrimalEnergy;

   std::cout<<" Non-smooth dual energy ";

   if (sparseFlag_) {
    std::cout<<"Not efficiently computable yet!";
   }
   else {
    std::cout<<compNonSmoothDualEnergy();
   }

   std::cout<<std::endl;

   std::cout<<"Computing energies took "<<myUtils::getTime() - tEnergy<<" seconds."<<std::endl;
   std::cout<<"ITERATION "<<cntIter<<" took "<<(tEnergy - tFull)<<" seconds."<<std::endl;

   tCompEnergies += myUtils::getTime() - tEnergy;
  }

  std::cout<<"ITERATION "<<cntIter<<" took "<<(myUtils::getTime() - tFull)<<" seconds."<<std::endl;
#endif
 } //MAIN TRN LOOP

 std::cout<<"solveNewton: total time taken "<<myUtils::getTime() - timeStart_<<" seconds."<<std::endl;

 //recoverFracPrimal();
 //recoverFeasPrimal();
 //setFracAsFeas();
 //recoverMaxPrimal(primalConsist_);

 std::cout<<std::fixed;
 std::cout<<std::setprecision(6);

 //std::cout<<"solveNewton: Fractional primal energy: "<<compSmoothPrimalEnergy()<<std::endl;
 std::cout<<"solveNewton: Smooth dual energy: ";
 if (sparseFlag_) {
  std::cout<<compEnergySparse(dualVar_)<<std::endl;
 }
 else {
  std::cout<<compEnergy(dualVar_)<<std::endl;
 }
 std::cout<<"solveNewton: Integral primal energy (feasible): "<<compIntPrimalEnergy()<<std::endl;

 recoverNodeFracPrimal();
 recoverMaxPrimal(primalFrac_);

 std::cout<<"solveNewton: Integral primal energy (directly from dual): "<<compIntPrimalEnergy()<<std::endl;
 std::cout<<" Best integral primal energy "<<bestPrimalEnergy<<std::endl;

 std::cout<<"solveNewton: Non-smooth dual energy: "<<compNonSmoothDualEnergy()<<std::endl;
 recoverFracPrimal();
 recoverFeasPrimal();
 std::cout<<"solveNewton: Non-smooth primal energy:"<<compNonSmoothPrimalEnergy()<<std::endl;

 std::cout<<"Total time spent on computing energies in each iteration: "<<tCompEnergies<<std::endl;

 return 0;
} //solveNewton()

int dualSys::solveQuasiNewton()
{
 quasiNewton quasiNewtonSolver(20); //Quasi Newton memory of 20

 precondFlag_ = 1; //0: Diagonal; 1: Block diagonal; 2: Incomplete Cholesky; 3: Quasi-Newton.

 if (precondFlag_ == 2) { //create Incomplete Cholesky data-structures
  inCholDiag_ = new double[nDualVar_];
  inCholColPtr_ = new int[nDualVar_ + 1];
 }
 else if (precondFlag_ == 3) { //create quasi Newton data-structures
  sVec_ = new double[nDualVar_]();
  yVec_ = new double[nDualVar_]();
  resVec_ = new double[nDualVar_];
  prVec_ = new double[nDualVar_];

  numVarQN_ = nDualVar_;
  numPairQN_ = 20;
  iopQN_ = 1;
  workVecQN_ = new double[4*(numPairQN_+1)*numVarQN_ + 2*(numPairQN_+1) + 1](); //4(M+1)N + 2(M+1) + 1
  lwQN_ = 4*(numPairQN_+1)*numVarQN_ + 2*(numPairQN_+1) + 1;
  iWorkVecQN_ = new int[2*numPairQN_ + 30](); //2M + 30
  liwQN_ = 2*numPairQN_ + 30;
  buildQN_ = true;
 }

 double tCompEnergies = 0;
 bool contIter = true;
 int cntIter = 0, cntIterTau = 0;
 double gradBasedDamp = 0;
// bool gradDampFlag = true;

 dampLambda_ = 10; //####

 Eigen::VectorXd eigenGuess = Eigen::VectorXd::Zero(nDualVar_);

 while ((contIter) && (cntIter < maxIter_)) {
//  double tFull = myUtils::getTime();

  ++cntIter;
  ++cntIterTau;

  eigenHess_.reserve(reserveHess_);

//  double tGHE = myUtils::getTime(); //time to comp gradient, hessian and current energy

  if (popGradEnergy() == -1) {
   return -1;
  }

  double gradNorm = myUtils::norm<double>(gradient_, nDualVar_);

  std::cout<<"gradient norm is "<<gradNorm<<std::endl;

  if (cntIter == 1) {
   gradBasedDamp = gradNorm/gradDampFactor_;
  }

  if ((annealIval_ != -1) && (gradNorm < gradBasedDamp) && (tau_ < tauMax_)) {
   cntIterTau = 1;
   tau_ *= tauScale_;
//   gradDampFlag = true; //earlier, it was 3 ####
   dampLambda_ = 10; //####

   eigenHess_.setZero();
   eigenHess_.reserve(reserveHess_);

   if (popGradEnergy() == -1) {
    return -1;
   }

   gradNorm = myUtils::norm<double>(gradient_, nDualVar_);
   gradBasedDamp = gradNorm/gradDampFactor_;

   if (gradBasedDamp < dampTol_) {
    gradBasedDamp = dampTol_;
   }
  }

  int gradMaxInd = myUtils::argmaxAbs<double>(gradient_,0,nDualVar_);

  double gradMax = std::abs(gradient_[gradMaxInd]);

  std::cout<<"ITERATION "<<cntIter<<" starts, with tau "<<tau_<<" and Gradient threshold "<<gradBasedDamp<<"."<<std::endl;

  int checkIter = -1;

  std::string opName;

  if (cntIter == checkIter) {
   opName = "gradient_" + std::to_string(checkIter) + ".txt";

   std::ofstream opGrad(opName);
   opGrad<<std::scientific;
   opGrad<<std::setprecision(6);
   for (std::vector<double>::iterator gradIter = gradient_.begin(); gradIter != gradient_.end(); ++gradIter) {
    opGrad<<*gradIter<<std::endl;
   }
   opGrad.close();
  }

  if (cntIter == checkIter) {
   opName = "dual_" + std::to_string(checkIter) + ".txt";

   std::ofstream opDual(opName);
   opDual<<std::scientific;
   opDual<<std::setprecision(6);
   for (std::vector<double>::iterator dualIter = dualVar_.begin(); dualIter != dualVar_.end(); ++dualIter) {
    opDual<<*dualIter<<std::endl;
   }
   opDual.close();
  }

  std::cout<<"gradMax "<<gradMax<<" gradNorm "<<gradNorm<<" gradTol "<<gradTol_<<std::endl;

  if ((gradMax > gradTol_) && (gradNorm > gradTol_)) {
   Eigen::VectorXd eigenStep(nDualVar_);

   std::ofstream hessFile;

   if (cntIter == checkIter) {
    opName = "hessian_" + std::to_string(checkIter) + ".txt";

    hessFile.open(opName);
    hessFile<<std::scientific;
    hessFile<<std::setprecision(6);

    for (int cJ = 0; cJ != nCliq_; ++cJ) {
     int sizCliqJ = subProb_[cJ].sizCliq_;
     int offsetOne = subProb_[cJ].getCliqOffset();
     std::set<int> cliqNeigh = subProb_[cJ].cliqNeigh_;

     std::vector<int> nodeOffsetJ = subProb_[cJ].getNodeOffset();
     std::vector<short> nLabelJ = subProb_[cJ].getNodeLabel();

     for (int nJ = 0; nJ != sizCliqJ; ++nJ) {
      for (int lJ = 0; lJ != nLabelJ[nJ]; ++lJ) {
       int curJ = offsetOne + nodeOffsetJ[nJ] + lJ;

       for (std::set<int>::iterator cI = cliqNeigh.begin(); cI != cliqNeigh.end(); ++cI) {
        std::vector<int> nodeOffsetI = subProb_[*cI].getNodeOffset();
        std::vector<short> nLabelI = subProb_[*cI].getNodeLabel();

        for (int nI = 0; nI != subProb_[*cI].sizCliq_; ++nI) {
         for (int lI = 0; lI != nLabelI[nI]; ++lI) {
          int curI = subProb_[*cI].getCliqOffset() + nodeOffsetI[nI] + lI;

          hessFile<<curI<<" "<<curJ<<" "<<eigenHess_.coeffRef(curI,curJ)<<std::endl;
         }
        }
       }
      }
     }
    }

    hessFile.close();
   }

   Eigen::VectorXd oriGrad(nDualVar_);
   Eigen::VectorXd eigenDualVar(nDualVar_);

   for (std::size_t i = 0; i < nDualVar_; ++i) {
    oriGrad[i] = gradient_[i];
    eigenGrad_[i] = gradient_[i];
    eigenDualVar[i] = dualVar_[i];
   }

   double tQN = myUtils::getTime();

   eigenStep = quasiNewtonSolver.solve(eigenGrad_, eigenDualVar, 0, cntIterTau);

   for (std::size_t i = 0; i != nDualVar_; ++i) {
    if (isnan(eigenStep[i])) {
     std::cout<<"an element of quasi newton step is nan"<<std::endl;
     return -1;
    }
   }

   std::cout<<"QN iteration took "<<myUtils::getTime() - tQN<<" seconds."<<std::endl;

   if (precondFlag_ == 2) { //Incomplete Cholesky Preconditioner data structures have to be freed
    delete[] inCholLow_;
    delete[] inCholDiag_;
    delete[] inCholColPtr_;
    delete[] inCholRowInd_;
   }
   else if (precondFlag_ == 3) { //Quasi-Newton preconditioner data structures have to be freed
    if (cntIterTau % 10 == 0) {
     delete[] sVec_;
     delete[] yVec_;
     delete[] resVec_;
     delete[] prVec_;
    }
   }

   for (std::size_t i = 0; i != nDualVar_; ++i) {
    newtonStep_[i] = eigenStep[i];
   }

   Eigen::VectorXd eigenInter = eigenHess_.selfadjointView<Eigen::Lower>()*eigenStep;

   double interValOne = eigenStep.dot(eigenInter);

   interValOne *= 0.5;

   double interValTwo = eigenStep.dot(oriGrad);

   double nxtApproxEnergyDiff = interValOne + interValTwo; //diff. wrt current energy

   std::vector<double> z(nDualVar_);

   myUtils::addArray<double>(dualVar_,newtonStep_,z,nDualVar_);

   double nxtEnergy = (sparseFlag_) ? compEnergySparse(z):compEnergy(z);

   double rho = (nxtEnergy - curEnergy_)/nxtApproxEnergyDiff;

   if ((isnan(rho)) || (isinf(rho))) {
    std::cout<<"rho debug: nxtApproxEnergyDiff "<<nxtApproxEnergyDiff<<" nxtEnergy "<<nxtEnergy<<" curEnergy_ "<<curEnergy_<<std::endl;
   }

   double eta_1 = 0.25, eta_2 = 0.5, eta_3 = 0.9, sigma_1 = 2, sigma_2 = 0.5, sigma_3 = 0.25, eta_0 = 0.0001;

   //updating damping lambda inspired by "Newton's method for large-scale optimization" by Bouaricha et al

   //std::cout<<"next energy "<<nxtEnergy<<" "<<" current energy "<<curEnergy_<<" next approx energy diff "<<nxtApproxEnergyDiff<<" interValOne "<<interValOne<<" interValTwo "<<interValTwo<<std::endl;

    double stepNormOld = 0, stepNormNew = 0;
//    int newtonMaxInd = myUtils::argmaxAbs<double>(newtonStep_,0,nDualVar_);
//    std::cout<<"solveNewton: before line search newton step l-infinity norm: "<<newtonStep_[newtonMaxInd]<<". Euclidean norm: "<<newtonStepNorm<<std::endl;

//   double tLS = myUtils::getTime(); //time to perform line search

   int lsCnt = -1;

   rho = 0; //always perform line search ####

   if (rho <= eta_0) {
    stepNormOld = myUtils::norm<double>(newtonStep_, nDualVar_);

    std::vector<double> oriNewton = newtonStep_;
//    lsCnt = performCGBacktrack(stepBackVec);

//    std::vector<double> resetStep = newtonStep_;

//    newtonStep_ = oriNewton;
    lsCnt = performLineSearch();
    std::cout<<"finished line search "<<lsCnt<<" times."<<std::endl;

    std::cout<<"Newton step is ";
    for (std::size_t i = 0; i != newtonStep_.size(); ++i) {
     std::cout<<newtonStep_[i]<<" ";
    }
    std::cout<<std::endl;

//    newtonStep_ = resetStep;

    stepNormNew = myUtils::norm<double>(newtonStep_, nDualVar_);

    if (stepNormOld/stepNormNew > 8) {
     dampLambda_ *= pow(sigma_1,2);
    }
    else if (stepNormOld/stepNormNew > 4) {
     dampLambda_ *= pow(sigma_1,2);
    }
    else {
     dampLambda_ *= sigma_1;
    }
   }
   else if (rho <= eta_1) {
    dampLambda_ *= sigma_1;
   }
   else if ((rho > eta_1) && (rho <= eta_2)) {
    dampLambda_ = dampLambda_;
   }
   else if ((rho > eta_2) && (rho <= eta_3)) {
    dampLambda_ *= sigma_2;
   }
   else if (rho > eta_3) {
    dampLambda_ *= sigma_3;
   }

//   std::cout<<"trust region param rho "<<rho<<" updated lambda "<<dampLambda_<<std::endl;

//   lsCnt = performLineSearch();

//  int newtonMaxInd = myUtils::argmaxAbs<double>(newtonStep_,0,nDualVar_);
//  std::cout<<"solveNewton: after line search newton step l-infinity norm: "<<newtonStep_[newtonMaxInd]<<". Euclidean norm: "<<newtonStepNorm<<std::endl;
//   std::cout<<"solveNewton: performed line search "<<lsCnt<<" times, "<<(myUtils::getTime() - tLS)<<" seconds."<<std::endl;

   myUtils::addArray<double>(dualVar_,newtonStep_,dualVar_,nDualVar_);

//   int dualVarInd = myUtils::argmaxAbs<double>(dualVar_,0,nDualVar_);
//   double dualNorm = myUtils::norm<double>(dualVar_, nDualVar_);
//   std::cout<<"solveNewton: dual l-infinity norm: "<<dualVar_[dualVarInd]<<". Euclidean norm: "<<dualNorm<<std::endl;

   distributeDualVars();

   eigenHess_.setZero(); //change logic to insert in first iteration and coeffRef in subsequent ones.
  }
  else if ((tau_ == tauMax_) || (annealIval_ == -1)) {
   contIter = false;

   gradNorm = myUtils::norm<double>(gradient_, nDualVar_);
   gradMaxInd = myUtils::argmaxAbs<double>(gradient_,0,nDualVar_);
   gradMax = std::abs(gradient_[gradMaxInd]);

//   std::cout<<"solveNewton: gradient l-infinity norm: "<<gradMax<<", Euclidean norm: "<<gradNorm<<". Energy: "<<curEnergy_<<std::endl;
  }
  else {
   gradBasedDamp = 2*gradNorm;

   if (gradBasedDamp < 0.000001) {
    gradBasedDamp = 0.000001;
   }

   eigenHess_.setZero();
  }

#if 1
 if (cntIter % 1 == 0) {
  double tEnergy = myUtils::getTime();

  recoverFracPrimal();
  //recoverFeasPrimal(); ????
  //setFracAsFeas();
  recoverMaxPrimal(primalFrac_);

  std::cout<<std::fixed;
  std::cout<<std::setprecision(6);

//  std::cout<<"Fractional primal energy: "<<compSmoothPrimalEnergy()<<" Smooth dual energy: "<<curEnergy_<<" Integral primal energy (feasible): "<<compIntPrimalEnergy();

  recoverMaxPrimal(primalFrac_);
//  std::cout<<" Integral primal energy (directly from dual): "<<compIntPrimalEnergy()<<" Non-smooth dual energy: "<<compNonSmoothDualEnergy()<<std::endl;

//  std::cout<<"Computing energies took "<<myUtils::getTime() - tEnergy<<" seconds."<<std::endl;
//  std::cout<<"NOTE: ITERATION "<<cntIter<<" took "<<(tEnergy - tFull)<<" seconds. The following figure includes unnecessary energy computation."<<std::endl;

  tCompEnergies += myUtils::getTime() - tEnergy;
 }
#endif

//  std::cout<<"ITERATION "<<cntIter<<" took "<<(myUtils::getTime() - tFull)<<" seconds.";
 }

 recoverFracPrimal();
 recoverFeasPrimal();
 recoverMaxPrimal(primalConsist_);

 std::cout<<std::fixed;
 std::cout<<std::setprecision(6);

 std::cout<<"solveQuasiNewton: Fractional primal energy: "<<compSmoothPrimalEnergy()<<std::endl;
 std::cout<<"solveQuasiNewton: Smooth dual energy: ";

 if (sparseFlag_) {
  std::cout<<compEnergySparse(dualVar_)<<std::endl;
 }
 else {
  std::cout<<compEnergy(dualVar_)<<std::endl;
 }
 std::cout<<"solveQuasiNewton: Integral primal energy (feasible): "<<compIntPrimalEnergy()<<std::endl;

 recoverMaxPrimal(primalFrac_);
 std::cout<<"solveQuasiNewton: Integral primal energy (directly from dual): "<<compIntPrimalEnergy()<<std::endl;
 std::cout<<"solveQuasiNewton: Non-smooth dual energy: "<<compNonSmoothDualEnergy()<<std::endl;

 std::cout<<"solveQuasiNewton: Total time spent on computing energies in each iteration: "<<tCompEnergies<<std::endl;

 return 0;
} //solveQuasiNewton

double dualSys::compEnergySparse(std::vector<double> var)
{
 double expVal = 0;
 double f1 = 0;

 double stgTerm = 0;

 if (stgCvxFlag_) {
  for (std::vector<double>::iterator varIter = var.begin(); varIter != var.end(); ++varIter) {
   stgTerm += (*varIter)*(*varIter);
  }

  stgTerm *= (stgCvxCoeff_/2);
 }

//#pragma omp parallel for reduction(+ : f1)
 for (int iCliq = 0; iCliq != nCliq_; ++iCliq) {
  double f1Energy = 0;

  std::vector<double> subProbVar(var.begin() + subProb_[iCliq].getCliqOffset(), var.begin() + subProb_[iCliq].getCliqOffset() + subProb_[iCliq].getDualSiz());

  cliqSPSparseEnergy(&subProb_[iCliq], tau_, subProbVar, f1Energy);

  f1 += f1Energy;
 }

// std::cout<<"DEBUG: F1 "<<f1<<std::endl;

 double f2 = 0;

//#pragma omp parallel for reduction(+:f2)
 for (int i = 0; i < nNode_; ++i) {
  int nLabel = nLabel_[i];
  double f2Node = 0;
  double f2Max = 0;
  double* f2Whole = new double[nLabel];
  for (int j = 0; j != nLabel; ++j) {
   double dualSum = 0;
   for (std::vector<int>::iterator k = cliqPerNode_[i].begin(); k != cliqPerNode_[i].end(); ++k) {
    int nodeInd = myUtils::findNodeIndex(subProb_[*k].memNode_, i);
    dualSum += var[subProb_[*k].getCliqOffset() + subProb_[*k].getNodeOffset()[nodeInd] + j];

//    if (stgCvxFlag_) {
//     stgTerm += 0.5*stgCvxCoeff_*pow(var[subProb_[*k].getCliqOffset() + subProb_[*k].getNodeOffset()[nodeInd] + j],2);
//    }
   }

   f2Whole[j] = uEnergy_[unaryOffset_[i] + j] + dualSum;
  }

  f2Max = *std::max_element(f2Whole, f2Whole + nLabel);

//  std::cout<<"DEBUG: F2MAX "<<f2Max<<std::endl;

  for (int j = 0; j != nLabel; ++j) {
   expVal = exp(tau_*(f2Whole[j] - f2Max));

   myUtils::checkRangeError(expVal);

//   std::cout<<"F2: EXPONENT "<<tau_*(f2Whole[j] - f2Max)<<" EXPVAL "<<expVal<<std::endl;

   f2Node += expVal;
  }

  f2 += (log(f2Node) + tau_*f2Max);

  delete[] f2Whole;
 }

 double returnVal = f1 + (1/tau_)*f2;

 if (stgCvxFlag_) {
  returnVal += stgTerm;
  std::cout<<"compEnergySparse: strong convexity term added to the energy: "<<stgTerm<<std::endl;
 }

 if (isinf(returnVal)) {
  std::cout<<"compEnergySparse: energy is INF"<<std::endl;
 }
 else if (isnan(returnVal)) {
  std::cout<<"compEnergySparse: energy is NAN!"<<std::endl;
 }

 return returnVal;
}

double dualSys::compEnergy(std::vector<double> var)
{
 double expVal = 0;
 double f1 = 0;

 for (int i = 0; i != nCliq_; ++i) {
  short sizCliq = subProb_[i].sizCliq_;
  std::vector<short> nLabel = subProb_[i].getNodeLabel();
  //std::vector<double> cEnergy = subProb_[i].getCE();
  //std::vector<double> cEnergy = cEnergy2DVec_[cliqOffset_[i]];
  std::vector<int> nodeOffset = subProb_[i].getNodeOffset();
  double f1Cliq = 0;
  double f1Max = 0;
  double* f1Whole = new double[subProb_[i].nCliqLab_];

  for (int j = 0; j != subProb_[i].nCliqLab_; ++j) {
   std::vector<short> cliqLab;

   double labPull = subProb_[i].nCliqLab_;

   for (int k = 0; k != sizCliq; ++k) {
    labPull /= nLabel[k];

    int labPullTwo = ceil((j+1)/labPull);

    if (labPullTwo % nLabel[k] == 0) {
     cliqLab.push_back(nLabel[k] - 1);
    }
    else {
     cliqLab.push_back((labPullTwo % nLabel[k]) - 1);
    }
   }

   double dualSum = 0;

   for (int k = 0; k != sizCliq; ++k) {
    dualSum += var[subProb_[i].getCliqOffset() + nodeOffset[k] + cliqLab[k]];
    if (isnan(dualSum)) {
     std::cout<<"F1 DUAL SUM IS NAN! dual variable is "<<var[subProb_[i].getCliqOffset() + nodeOffset[k] + cliqLab[k]]<<std::endl;
    }
   }

   f1Whole[j] = subProb_[i].getCE(j) - dualSum;

//   std::cout<<"DEBUG: F1WHOLE "<<f1Whole[j]<<std::endl;
  }

  f1Max = *std::max_element(f1Whole,f1Whole + subProb_[i].nCliqLab_);

//  std::cout<<"DEBUG: F1MAX "<<f1Max<<" NCLIQLAB "<<subProb_[i].nCliqLab_<<std::endl;

  for (int j = 0; j != subProb_[i].nCliqLab_; ++j) {
   expVal =  exp(tau_*(f1Whole[j] - f1Max));

   myUtils::checkRangeError(expVal);

//   std::cout<<"F1: EXPONENT "<<tau_*(f1Whole[j] - f1Max)<<" EXPVAL "<<expVal<<std::endl;

   f1Cliq += expVal;
  }

  f1 += (log(f1Cliq) + tau_*f1Max);

  delete[] f1Whole;
 }

// std::cout<<"DEBUG: F1 "<<f1<<std::endl;

 double f2 = 0;

 double stgTerm = 0;

 for (int i = 0; i != nNode_; ++i) {
  int nLabel = nLabel_[i];
  double f2Node = 0;
  double f2Max = 0;
  double* f2Whole = new double[nLabel];
  for (int j = 0; j != nLabel; ++j) {
   double dualSum = 0;
   for (std::vector<int>::iterator k = cliqPerNode_[i].begin(); k != cliqPerNode_[i].end(); ++k) {
    int nodeInd = myUtils::findNodeIndex(subProb_[*k].memNode_, i);
    dualSum += var[subProb_[*k].getCliqOffset() + subProb_[*k].getNodeOffset()[nodeInd] + j];

    if (stgCvxFlag_) {
     stgTerm += 0.5*stgCvxCoeff_*pow(var[subProb_[*k].getCliqOffset() + subProb_[*k].getNodeOffset()[nodeInd] + j],2);
    }
   }

   f2Whole[j] = uEnergy_[unaryOffset_[i] + j] + dualSum;
  }

  f2Max = *std::max_element(f2Whole, f2Whole + nLabel);

//  std::cout<<"DEBUG: F2MAX "<<f2Max<<std::endl;

  for (int j = 0; j != nLabel; ++j) {
   expVal = exp(tau_*(f2Whole[j] - f2Max));

   myUtils::checkRangeError(expVal);

//   std::cout<<"F2: EXPONENT "<<tau_*(f2Whole[j] - f2Max)<<" EXPVAL "<<expVal<<std::endl;

   f2Node += expVal;
  }

  f2 += (log(f2Node) + tau_*f2Max);

  delete[] f2Whole;
 }

 double returnVal = (1/tau_)*(f1 + f2);

 if (stgCvxFlag_) {
  returnVal += stgTerm;
  //std::cout<<"compEnergy: strong convexity term added to the energy: "<<stgTerm<<std::endl;
 }

 std::cout<<"compEnergy: f1 "<<(1/tau_)*f1<<" f2 "<<(1/tau_)*f2<<std::endl;

 if (isinf(returnVal)) {
  std::cout<<"compEnergy: energy is INF"<<std::endl;
 }
 else if (isnan(returnVal)) {
  std::cout<<"compEnergy: energy is NAN!"<<std::endl;
 }

 return returnVal;
}

double dualSys::compNonSmoothDualEnergy()
{
 double f1 = 0;

 for (int i = 0; i != nCliq_; ++i) {
  //std::vector<double> cEnergy = subProb_[i].getCE();
  //std::vector<double> cEnergy = cEnergy2DVec_[cliqOffset_[i]];
  std::vector<int> nodeOffset = subProb_[i].getNodeOffset();
  std::vector<short> nLabel = subProb_[i].getNodeLabel();
  int sizCliq = subProb_[i].sizCliq_;

  double f1Cliq = -1.0*pow(10,10);

  for (int j = 0; j != subProb_[i].nCliqLab_; ++j) {
   double dualSum = 0;

   std::vector<short> cliqLab;

   double labPull = subProb_[i].nCliqLab_;

   for (int k = 0; k != sizCliq; ++k) {
    labPull /= nLabel[k];

    int labPullTwo = ceil((j+1)/labPull);

    if (labPullTwo % nLabel[k] == 0) {
     cliqLab.push_back(nLabel[k] - 1);
    }
    else {
     cliqLab.push_back((labPullTwo % nLabel[k]) - 1);
    }
   }

   for (int k = 0; k != sizCliq; ++k) {
    dualSum += subProb_[i].getDualVar()[nodeOffset[k] + cliqLab[k]];
   }

   if ((subProb_[i].getCE(j) - dualSum) > f1Cliq) {
    f1Cliq = subProb_[i].getCE(j) - dualSum;
   }
  }

  f1 += f1Cliq;
 }

 double f2 = 0;
 for (int i = 0; i != nNode_; ++i) {
  int nLabel = nLabel_[i];
  double f2Node = -1.0*pow(10,10);

  for (int j = 0; j != nLabel; ++j) {
   double dualSum = 0;
   for (std::vector<int>::iterator k = cliqPerNode_[i].begin(); k != cliqPerNode_[i].end(); ++k) {
    dualSum += subProb_[*k].getDualVar()[subProb_[*k].getNodeOffset()[myUtils::findNodeIndex(subProb_[*k].memNode_, i)] + j];
   }

   if ((uEnergy_[unaryOffset_[i] + j] + dualSum) > f2Node) {
    f2Node = uEnergy_[unaryOffset_[i] + j] + dualSum;
   }
  }

  f2 += f2Node;
 }

 return (f1 + f2);
}

void dualSys::recoverFracPrimal()
{
 double expVal, fracVal;

#if 1
 for (int i = 0; i != nNode_; ++i) {
  int nLabel = nLabel_[i];

  double argMax = 0;

  std::vector<double> argVal(nLabel);

  for (int j = 0; j != nLabel; ++j) {
   double dualSum = 0;

    for (std::vector<int>::iterator k = cliqPerNode_[i].begin(); k != cliqPerNode_[i].end(); ++k) {
     dualSum += subProb_[*k].getDualVar()[subProb_[*k].getNodeOffset()[myUtils::findNodeIndex(subProb_[*k].memNode_, i)] + j];
    }

   argVal[j] = uEnergy_[unaryOffset_[i] + j] + dualSum;

   if (j == 0) {
    argMax = argVal[j];
   }
   else {
    if (argVal[j] > argMax) {
     argMax = argVal[j];
    }
   }
  }

  double nodeNorm = 0;

  for (int j = 0; j != nLabel; ++j) {
   expVal = exp(tau_*(argVal[j] - argMax));
   myUtils::checkRangeError(expVal);

   primalFrac_[unaryOffset_[i] + j] = expVal;

   nodeNorm += expVal;
  }

  for (int j = 0; j != nLabel; ++j) {
   if (nodeNorm == 0) { //all of primalFrac_ is zero
    fracVal = 0;
   }
   else {
    fracVal = (1/nodeNorm)*primalFrac_[unaryOffset_[i] + j];
   }

   if (fracVal < minNumAllow_) {
    primalFrac_[unaryOffset_[i] + j] = 0;
   }
   else {
    primalFrac_[unaryOffset_[i] + j] = fracVal;
   }
  }
 }
#endif

 for (int i = 0; i != nCliq_; ++i) {
  double argMax = 0;
  int nCliqLab = subProb_[i].nCliqLab_;
  std::vector<double> argVal(nCliqLab);
  std::vector<short> nLabel = subProb_[i].getNodeLabel();
  std::vector<int> nodeOffset = subProb_[i].getNodeOffset();
  int sizCliq = subProb_[i].sizCliq_;
  std::vector<int> memNode = subProb_[i].memNode_;
  int subDualSiz = subProb_[i].getDualSiz();

  for (int j = 0; j != nCliqLab; ++j) {
   double dualSum = 0;

   double labPull = nCliqLab;

   for (int k = 0; k != sizCliq; ++k) {
    short cliqLabCur;
    labPull /= nLabel[k];

    int labPullTwo = ceil((j+1)/labPull);

    if (labPullTwo % nLabel[k] == 0) {
     cliqLabCur = nLabel[k] - 1;
    }
    else {
     cliqLabCur = (labPullTwo % nLabel[k]) - 1;
    }

    dualSum += subProb_[i].getDualVar()[nodeOffset[k] + cliqLabCur];
   }

   //argVal[j] = subProb_[i].getCE()[j] - dualSum;
   //argVal[j] = cEnergy2DVec_[cliqOffset_[i]][j] - dualSum;
   argVal[j] = subProb_[i].getCE(j) - dualSum;

   if (j == 0) {
    argMax = argVal[j];
   }
   else {
    if (argVal[j] > argMax) {
     argMax = argVal[j];
    }
   }
  } //for j = [0:nCliqLab)

  double cliqNorm = 0;

  subProb_[i].primalCliqFrac_.resize(nCliqLab);

  for (int j = 0; j != nCliqLab; ++j) {
   expVal = exp(tau_*(argVal[j] - argMax));
   myUtils::checkRangeError(expVal);
   subProb_[i].primalCliqFrac_[j] = expVal;
   cliqNorm += expVal;
  }

  std::vector<double> curMargSum(subDualSiz, 0);

  for (int j = 0; j != nCliqLab; ++j) {
   fracVal =  (1/cliqNorm)*subProb_[i].primalCliqFrac_[j];


   if (fracVal < minNumAllow_) {
    subProb_[i].primalCliqFrac_[j] = 0;
   }
   else {
    subProb_[i].primalCliqFrac_[j] = fracVal;
   }

#if 0
   std::vector<short> cliqLab;
   double labPull = nCliqLab;

   for (int k = 0; k != sizCliq; ++k) {
    labPull /= nLabel[k];

    int labPullTwo = ceil((j+1)/labPull);

    if (labPullTwo % nLabel[k] == 0) {
     cliqLab.push_back(nLabel[k] - 1);
    }
    else {
     cliqLab.push_back((labPullTwo % nLabel[k]) - 1);
    }
   }

   for (int k = 0; k != sizCliq; ++k) {
    for (int l = 0; l != nLabel[k]; ++l) {
     if (cliqLab[k] == l) {
      curMargSum[nodeOffset[k] + l] += subProb_[i].primalCliqFrac_[j];
     }
    }
   }
#endif
  }

#if 0
  for (int iNode = 0; iNode != sizCliq; ++iNode) {
   for (int iLabel = 0; iLabel != nLabel[iNode]; ++iLabel) {
     primalFrac_[unaryOffset_[memNode[iNode]] + iLabel] = curMargSum[nodeOffset[iNode] + iLabel];
   }
  }
#endif
 } //for iCliq
}

void dualSys::recoverNodeFracPrimal()
{
 double expVal, fracVal;

 for (int i = 0; i != nNode_; ++i) {
  int nLabel = nLabel_[i];

  double argMax = 0;

  std::vector<double> argVal(nLabel);

  for (int j = 0; j != nLabel; ++j) {
   double dualSum = 0;

    for (std::vector<int>::iterator k = cliqPerNode_[i].begin(); k != cliqPerNode_[i].end(); ++k) {
     dualSum += subProb_[*k].getDualVar()[subProb_[*k].getNodeOffset()[myUtils::findNodeIndex(subProb_[*k].memNode_, i)] + j];
    }

   argVal[j] = uEnergy_[unaryOffset_[i] + j] + dualSum;

   if (j == 0) {
    argMax = argVal[j];
   }
   else {
    if (argVal[j] > argMax) {
     argMax = argVal[j];
    }
   }
  }

  double nodeNorm = 0;

  for (int j = 0; j != nLabel; ++j) {
   expVal = exp(tau_*(argVal[j] - argMax));
   myUtils::checkRangeError(expVal);

   primalFrac_[unaryOffset_[i] + j] = expVal;

   nodeNorm += expVal;
  }

  for (int j = 0; j != nLabel; ++j) {
   if (nodeNorm == 0) { //all of primalFrac_ is zero
    fracVal = 0;
   }
   else {
    fracVal = (1/nodeNorm)*primalFrac_[unaryOffset_[i] + j];
   }

   primalFrac_[unaryOffset_[i] + j] = fracVal;
  }
 }
}

void dualSys::recoverMaxPrimal(std::vector<double> nodeFrac)
{
 for (int i = 0; i != nNode_; ++i) {
  int maxInd = std::distance(nodeFrac.begin() + unaryOffset_[i], std::max_element(nodeFrac.begin() + unaryOffset_[i], nodeFrac.begin() + unaryOffset_[i]+nLabel_[i]));
//  int maxInd = myUtils::argmax<double>(nodeFrac, i*nLabel_, i*nLabel_ + nLabel_);
  primalMax_[i] = maxInd;
 }

 for (int i = 0; i != nCliq_; ++i) {
  int cliqInd = 0;
  for (int j = 0; j != subProb_[i].sizCliq_; ++j) {
   int curNode = subProb_[i].memNode_[j];
   int nLabel = nLabel_[curNode];
   cliqInd += primalMax_[curNode]*pow(nLabel,(subProb_[i].sizCliq_ - 1 - j));
  }

  subProb_[i].primalCliqMax_ = cliqInd;
 }
}

void dualSys::assignPrimalVars(std::string labelFile)
{
 std::ifstream lbFile(labelFile);
 std::cout<<"label file read status "<<lbFile.is_open()<<std::endl;
 std::istringstream sin;
 std::string line;
 std::getline(lbFile,line);
 sin.str(line);

 for (int i = 0; i != nNode_; ++i) {
  sin>>primalMax_[i];
 }

 lbFile.close();

 for (int i = 0; i != nCliq_; ++i) {
  int cliqInd = 0;

  for (int j = 0; j != subProb_[i].sizCliq_; ++j) {
   int curNode = subProb_[i].memNode_[j];
   int nLabel = nLabel_[curNode];
   cliqInd += primalMax_[curNode]*pow(nLabel,(subProb_[i].sizCliq_ - 1 - j));
  }
  subProb_[i].primalCliqMax_ = cliqInd;
 }

}

void dualSys::recoverFeasPrimal()
{
 //NOTE: THIS APPROACH REQUIRES TOLERANCE BASED THRESHOLDS FOR ACHIEVING ZERO DUALITY GAP TOWARDS THE OPTIMUM
 //THESE ADJUSTMENTS ARE INDICATED BY ^^...^^

 primalConsist_.clear();
 primalConsist_.resize(uEnergy_.size());

 std::vector<std::vector<double> > cliqMargSum(nCliq_);

 for (int iCliq = 0; iCliq != nCliq_; ++iCliq) {
  int sizCliq = subProb_[iCliq].sizCliq_;
  int subDualSiz = subProb_[iCliq].getDualSiz();
  std::vector<int> nodeOffset = subProb_[iCliq].getNodeOffset();
  std::vector<short> nLabel = subProb_[iCliq].getNodeLabel();
  int nCliqLab = subProb_[iCliq].nCliqLab_;

  std::vector<double> curMargSum(subDualSiz, 0);

  for (int j = 0; j != nCliqLab; ++j) {
   std::vector<short> cliqLab;
   double labPull = nCliqLab;

   for (int k = 0; k != sizCliq; ++k) {
    labPull /= nLabel[k];

    int labPullTwo = ceil((j+1)/labPull);

    if (labPullTwo % nLabel[k] == 0) {
     cliqLab.push_back(nLabel[k] - 1);
    }
    else {
     cliqLab.push_back((labPullTwo % nLabel[k]) - 1);
    }
   }

   for (int k = 0; k != sizCliq; ++k) {
    for (int l = 0; l != nLabel[k]; ++l) {
     if (cliqLab[k] == l) {
      curMargSum[nodeOffset[k] + l] += subProb_[iCliq].primalCliqFrac_[j];
     }
    }
   }
  }
  cliqMargSum[iCliq] = curMargSum;

  std::vector<int> memNode = subProb_[iCliq].memNode_;

 }//for iCliq

 for (int iNode = 0; iNode != nNode_; ++iNode) {
  int nLabel = nLabel_[iNode];

  for (int iLabel = 0; iLabel != nLabel; ++iLabel) {

   double cliqSum = 0;
   double cliqDiv = 0;

   for (std::vector<int>::iterator k = cliqPerNode_[iNode].begin(); k != cliqPerNode_[iNode].end(); ++k) {
    int nodeInd = myUtils::findNodeIndex(subProb_[*k].memNode_, iNode);

    std::vector<short> cliqNodeLabel = subProb_[*k].getNodeLabel();

    double margNorm = 1;

    for (int l = 0; l != subProb_[*k].sizCliq_; ++l) {
     if (l != nodeInd) {
      margNorm *= cliqNodeLabel[l];
     }
    }
    margNorm = 1/margNorm;

    cliqSum += margNorm*cliqMargSum[*k][subProb_[*k].getNodeOffset()[nodeInd] + iLabel];
    cliqDiv += margNorm;
   }

   primalConsist_[unaryOffset_[iNode] + iLabel] = (1.0/(1.0 + cliqDiv))*(primalFrac_[unaryOffset_[iNode] + iLabel] + cliqSum);
  } //for iLabel
 } //for iNode

 double lambd = 0.0;

 for (int i = 0; i != nCliq_; ++i) {
  int sizCliq = subProb_[i].sizCliq_;
  std::vector<int> nodeOffset = subProb_[i].getNodeOffset();
  std::vector<short> nLabel = subProb_[i].getNodeLabel();
  int nCliqLab = subProb_[i].nCliqLab_;

  subProb_[i].primalCliqConsist_.resize(nCliqLab);

  for (int j = 0; j != nCliqLab; ++j) {

   std::vector<short> cliqLab;
   double labPull = nCliqLab;

   for (int k = 0; k != sizCliq; ++k) {
    labPull /= nLabel[k];

    int labPullTwo = ceil((j+1)/labPull);

    if (labPullTwo % nLabel[k] == 0) {
     cliqLab.push_back(nLabel[k] - 1);
    }
    else {
     cliqLab.push_back((labPullTwo % nLabel[k]) - 1);
    }
   }

   double nodeSum = 0.0;
   for (int k = 0; k != sizCliq; ++k) {

    double margNorm = 1;
    for (int l = 0; l != sizCliq; ++l) {
     if (l != k) {
      margNorm *= nLabel[l];
     }
    }
    margNorm = 1/margNorm;

    nodeSum += margNorm*(cliqMargSum[i][nodeOffset[k] + cliqLab[k]] - primalConsist_[unaryOffset_[subProb_[i].memNode_[k]] + cliqLab[k]]);
   }

   subProb_[i].primalCliqConsist_[j] = subProb_[i].primalCliqFrac_[j] - nodeSum;

   double denTerm = subProb_[i].primalCliqConsist_[j] - (1.0/subProb_[i].nCliqLab_);

   if (subProb_[i].primalCliqConsist_[j] < -1*pow(10,-5)) { //^^MODIFY LAMBDA FOR ONLY SUFFICIENTLY LARGE VALUES^^
    if (lambd < subProb_[i].primalCliqConsist_[j]/denTerm) {
     lambd = subProb_[i].primalCliqConsist_[j]/denTerm;
    }
   }
   else if (subProb_[i].primalCliqConsist_[j] - 1 > pow(10,-5)) { //^^MODIFY LAMBDA FOR ONLY SUFFICIENTLY LARGE VALUES^^
    if (lambd < (subProb_[i].primalCliqConsist_[j] - 1.0)/denTerm) {
     lambd = (subProb_[i].primalCliqConsist_[j] - 1.0)/denTerm;
    }
   }
  } //for j
 } //for i

 for (int i = 0; i != nNode_; ++i) {
  int nLabel = nLabel_[i];
  double nodeLambdTerm = lambd/nLabel;

  for (int j = 0; j != nLabel; ++j) {
   primalConsist_[unaryOffset_[i] + j] = (1.0 - lambd)*primalConsist_[unaryOffset_[i] + j] + nodeLambdTerm;
  }
 }

 for (int i = 0; i != nCliq_; ++i) {
  double cliqLambdTerm = lambd/subProb_[i].nCliqLab_;
  int subDualSiz = subProb_[i].getDualSiz();
  std::vector<short> nLabel = subProb_[i].getNodeLabel();
  std::vector<int> nodeOffset = subProb_[i].getNodeOffset();
  std::vector<int> memNode = subProb_[i].memNode_;
  int nCliqLab = subProb_[i].nCliqLab_;

  std::vector<double> curMargSum(subDualSiz, 0);

  for (int j = 0; j != nCliqLab; ++j) {
   subProb_[i].primalCliqConsist_[j] = (1.0 - lambd)*subProb_[i].primalCliqConsist_[j] + cliqLambdTerm;
  }

 } //for i
}

double dualSys::compIntPrimalEnergy()
{
 double energy = 0;
 for (int i = 0; i != nCliq_; ++i) {
  energy += subProb_[i].getCE(subProb_[i].primalCliqMax_);
 }

 for (int i = 0; i != nNode_; ++i) {
  energy += uEnergy_[unaryOffset_[i] + primalMax_[i]];
 }

 return energy;
}

double dualSys::compSmoothPrimalEnergy()
{
 double energy = 0;
 double entropyTerm, logVal;

 int probCliqCnt = 0, probNodeCnt = 0;

 for (int i = 0; i != nCliq_; ++i) {
  int nCliqLab = subProb_[i].nCliqLab_;
  for (int j = 0; j != nCliqLab; ++j) {
   logVal = log(subProb_[i].primalCliqConsist_[j]);

   if ((errno == ERANGE) || (isnan(logVal))) {
    entropyTerm = 0;
    errno = 0;
   }
   else {
    entropyTerm = -1*subProb_[i].primalCliqConsist_[j]*logVal;
   }

   energy += subProb_[i].getCE(j)*subProb_[i].primalCliqConsist_[j] + (1/tau_)*entropyTerm;
   if ((isnan(energy)) && (probCliqCnt == 0)) {
    ++probCliqCnt;
   }
  } //for j
 } //for i

 for (int i = 0; i != nNode_; ++i) {
  int nLabel = nLabel_[i];

  for (int j = 0; j != nLabel; ++j) {
   logVal = log(primalConsist_[unaryOffset_[i] + j]);

   if ((errno == ERANGE) || (isnan(logVal))) {
    entropyTerm = 0;
    errno = 0;
   }
   else {
    entropyTerm = -1*primalConsist_[unaryOffset_[i] + j]*logVal;
   }

   energy += uEnergy_[unaryOffset_[i] + j]*primalConsist_[unaryOffset_[i] + j] + (1/tau_)*entropyTerm;
   if ((isnan(energy)) && (probNodeCnt == 0)) {
    ++probNodeCnt;
   }
  }
 }

 return energy;
}

double dualSys::compNonSmoothPrimalEnergy()
{
 double energy = 0;

 for (int i = 0; i != nCliq_; ++i) {
  int nCliqLab = subProb_[i].nCliqLab_;
  std::vector<double> primalCliqConsist = subProb_[i].primalCliqConsist_;

  for (int iCliqLab = 0; iCliqLab != nCliqLab; ++iCliqLab) {

   energy += subProb_[i].getCE(iCliqLab)*primalCliqConsist[iCliqLab];
   if (isnan(energy)) {
    std::cout<<"clique "<<i<<" cEnergy "<<subProb_[i].getCE(iCliqLab)<<" primalCliqConsist "<<primalCliqConsist[iCliqLab]<<std::endl;
   }
  }
 }

 for (int i = 0; i != nNode_; ++i) {
  int nLabel = nLabel_[i];

  for (int j = 0; j != nLabel; ++j) {
   energy += uEnergy_[unaryOffset_[i] + j]*primalConsist_[unaryOffset_[i] + j];
   if (isnan(energy)) {
    std::cout<<"node "<<i<<" uEnergy "<<uEnergy_[unaryOffset_[i] + j]<<" primalConsist "<<primalConsist_[unaryOffset_[i] + j]<<std::endl;
   }
  }
 }

 return energy;
}

int dualSys::distributeDualVars() {

 for (int i = 0; i != nCliq_; ++i) {
   std::copy(dualVar_.begin() + subProb_[i].getCliqOffset(), dualVar_.begin() + subProb_[i].getCliqOffset() + subProb_[i].getDualSiz(), subProb_[i].dualVar_.begin());
 }

 return 0;
}

int dualSys::distributeMomentum() {

 for (int i = 0; i != nCliq_; ++i) {
   std::copy(momentum_.begin() + subProb_[i].getCliqOffset(), momentum_.begin() + subProb_[i].getCliqOffset() + subProb_[i].getDualSiz(), subProb_[i].momentum_.begin());
 }

 return 0;
}

int dualSys::solveCG(Eigen::VectorXd& iterStep, std::vector<Eigen::VectorXd> &iterStepVec, int newtonIter)
{
 int cntIter = 1;

 iterStepVec.clear(); //populate fresh each time

 //data structures for inc. Cholesky
 int p = 100;
 int nzInChol = nNonZeroLower_ + p*nDualVar_;

 //data structures for Quasi Newton
#if 0
 int numVar, numPairs, iop;
 float *workVec;
 int *iWorkVec;
 int lw, liw;
#endif

 if (precondFlag_ == 2) { //create Incomplete Cholesky data-structures
  inCholLow_ = new double[nzInChol];
  inCholRowInd_ = new int[nzInChol];

//  double tInChol = myUtils::getTime();
  compInCholPrecond(p);
//  std::cout<<"computing incomplete Cholesky takes "<<myUtils::getTime()-tInChol<<std::endl;
 }
#if 0
 else if (precondFlag_ == 3) { //create quasi Newton data-structures
  numVar = nDualVar_;
  numPairs = 20;
  iop = 1;
  workVec = new float[4*(numPairs+1)*numVar + 2*(numPairs+1) + 10]; //4(M+1)N + 2(M+1) + 1
  lw = 4*(numPairs+1)*numVar + 2*(numPairs+1) + 1;
  iWorkVec = new int[2*numPairs + 30]; //2M + 30
  liw = 2*numPairs + 30;
  build = false;
 }
#endif

 Eigen::VectorXd b(nDualVar_), residual(nDualVar_), direc(nDualVar_), preconRes(nDualVar_);

// #pragma omp parallel for
// for (std::size_t i = 0; i < nDualVar_; ++i) {
//  iterStep[i] = 0;
// }

 b = -1*eigenGrad_;

 residual = getHessVecProd(iterStep) + eigenGrad_;

#if 0
 Eigen::VectorXd testVec(nDualVar_);

 for (int iDebug = 0; iDebug != nDualVar_; ++iDebug) {
  testVec.setZero(nDualVar_);

  testVec(iDebug) = 1;

  Eigen::VectorXd testFull = eigenHess_.selfadjointView<Eigen::Lower>()*testVec;
  Eigen::VectorXd testOptim = getOptimHessVecProd(testVec);

  for (int jDebug = 0; jDebug != nDualVar_; ++jDebug) {
   if ((testFull(jDebug) <= testOptim(jDebug) - pow(10,-6)) || (testFull(jDebug) >= testOptim(jDebug) + pow(10,-6))) {
    std::cout<<"SOMETHING WRONG! "<<testFull(jDebug)<<" "<<testOptim(jDebug)<<std::endl;
   }
  }
 }
 //debugEigenVec(testFull);
 //debugEigenVec(testOptim);

 for (int iDebug = 0 ; iDebug != nDualVar_; ++iDebug) {
  if ((residual[iDebug] <= residualTest[iDebug] - pow(10,-6)) || (residual[iDebug] >= residualTest[iDebug] + pow(10,-6))) {
   std::cout<<"SOMETHING WRONG!"<<std::endl;
  }
 }
#endif

 switch (precondFlag_)
 {
  case 0:
   preconRes = diagPrecond(residual);
   break;
  case 1:
   preconRes = blkJacobPrecond(residual);
   break;
  case 2:
   preconRes = applyInCholPrecond(residual);
   break;
  case 3:
   for (std::size_t i = 0; i != nDualVar_; ++i) {
    resVec_[i] = residual[i];
   }
   //precnd_(&numVarQN_, &numPairQN_, &iopQN_, &newtonIter, &cntIter, sVec_, yVec_, resVec_, prVec_, workVecQN_, &lwQN_, iWorkVecQN_, &liwQN_, &build, &info, &mssg);
   for (std::size_t i = 0; i != nDualVar_; ++i) {
    preconRes[i] = prVec_[i];
   }
   break;
  default:
   preconRes = residual;
   break;
 }

 direc = -1*preconRes;

 double resDotPreRes = residual.dot(preconRes);

 //forcing sequence is very critical; especially close to the optimum
 double forceTerm;

#if 0
 if (newtonIter < 25) {
  forceTerm = 0.5;
 }
 else if (newtonIter < 50) {
  forceTerm = 0.1;
 }
 else {
  forceTerm = 0.0001; //1.0/newtonIter;
 }
#else
 if (tau_ < tauMax_/4) {
  forceTerm = 0.01*sqrt(linSysScale_); //*(1.0/newtonIter); //0.0001
 }
 else if (tau_ < tauMax_/2) {
  forceTerm = 0.001*sqrt(linSysScale_); //*(1.0/newtonIter); //0.0001
 }
 else {
  forceTerm = 0.0001*sqrt(linSysScale_); //*(1.0/newtonIter); //0.0001
 }
#endif

 double residualCond = (forceTerm < sqrt(eigenGrad_.norm())) ? (forceTerm/sqrt(linSysScale_))*eigenGrad_.norm():sqrt(eigenGrad_.norm()/linSysScale_)*eigenGrad_.norm();

 if (residualCond < (1e-4)*linSysScale_) { //####
  residualCond = (1e-4)*linSysScale_;
 }

 std::cout<<"force term "<<forceTerm<<" sqrt(norm) "<<sqrt(eigenGrad_.norm())<<" residual condition "<<residualCond<<std::endl;

 bool exitCond = false;

 double totPrecondTime = 0;

// int backTrackPow = 1;
// double backTrackIndex = 1.3;

 while (!exitCond) {
  double alpha;
  double residualNorm = residual.norm();

  //std::cout<<"solveCG: iteration "<<cntIter<<" residual norm "<<residualNorm<<std::endl;

  if (cntIter == 250) {
   std::cout<<"solveCG: exit on max. no. of iterations. Residual "<<residualNorm<<". Returned step norm "<<iterStep.norm()<<". step l-infinity norm "<<iterStep.maxCoeff()<<". Iteration "<<cntIter<<std::endl;

   std::ifstream Infield("dump_hessian.txt");

   if (!Infield.good()) {
    std::ofstream dumpHessian("dump_hessian.txt", std::ofstream::trunc);
    std::ofstream dumpb("dump_b.txt", std::ofstream::trunc);

    for (int cliqJ = 0; cliqJ < nCliq_; ++cliqJ) {
     std::vector<short> nLabelJ = subProb_[cliqJ].getNodeLabel();
     int sizCliqJ = subProb_[cliqJ].sizCliq_; //OPTIMIZE - used only once
     int cliqOffsetJ = subProb_[cliqJ].getCliqOffset(); //OPTIMIZE - used only once
     std::vector<int> nodeOffsetJ = subProb_[cliqJ].getNodeOffset(); //OPTIMIZE - used only once
     std::set<int> cliqNeigh = subProb_[cliqJ].cliqNeigh_;

     for (int elemJ = 0; elemJ != sizCliqJ; ++elemJ) {
      for (int labelJ = 0; labelJ != nLabelJ[elemJ]; ++labelJ) {
       int curJ = cliqOffsetJ + nodeOffsetJ[elemJ] + labelJ;

       dumpb<<b[curJ]<<std::endl;

       for (std::set<int>::iterator cliqI = cliqNeigh.begin(); cliqI != cliqNeigh.end(); ++cliqI) {
        if (*cliqI >= cliqJ) {
         std::vector<short> nLabelI = subProb_[*cliqI].getNodeLabel();
         int sizCliqI = subProb_[*cliqI].sizCliq_; //OPTIMIZE - used only once
         int cliqOffsetI = subProb_[*cliqI].getCliqOffset(); //OPTIMIZE - used only once
         std::vector<int> nodeOffsetI = subProb_[*cliqI].getNodeOffset(); //OPTIMIZE - used only once

         for (int elemI = 0; elemI != sizCliqI; ++elemI) {
          for (int labelI = 0; labelI != nLabelI[elemI]; ++labelI) {

           int curI = cliqOffsetI + nodeOffsetI[elemI] + labelI;
           dumpHessian<<curI<<" "<<curJ<<" "<<eigenHess_.coeffRef(curI,curJ)<<std::endl;
          } //labelI
         } //elemI
        }
       } //for iterating through cliqNeigh
      } // for labelJ
     } //for elemJ
    } //for cliqJ

    dumpHessian.close();
    dumpb.close();
   }

   exitCond = true;
  }
  else if (residualNorm <= residualCond) {
   if (cntIter > 10) {

    std::ifstream Infield("dump_hessian.txt");

    if (!Infield.good()) {
     std::ofstream dumpHessian("dump_hessian.txt", std::ofstream::trunc);
     std::ofstream dumpb("dump_b.txt", std::ofstream::trunc);

     for (int cliqJ = 0; cliqJ < nCliq_; ++cliqJ) {
      std::vector<short> nLabelJ = subProb_[cliqJ].getNodeLabel();
      int sizCliqJ = subProb_[cliqJ].sizCliq_; //OPTIMIZE - used only once
      int cliqOffsetJ = subProb_[cliqJ].getCliqOffset(); //OPTIMIZE - used only once
      std::vector<int> nodeOffsetJ = subProb_[cliqJ].getNodeOffset(); //OPTIMIZE - used only once
      std::set<int> cliqNeigh = subProb_[cliqJ].cliqNeigh_;

      for (int elemJ = 0; elemJ != sizCliqJ; ++elemJ) {
       for (int labelJ = 0; labelJ != nLabelJ[elemJ]; ++labelJ) {
        int curJ = cliqOffsetJ + nodeOffsetJ[elemJ] + labelJ;

        dumpb<<b[curJ]<<std::endl;

        for (std::set<int>::iterator cliqI = cliqNeigh.begin(); cliqI != cliqNeigh.end(); ++cliqI) {
         if (*cliqI >= cliqJ) {
          std::vector<short> nLabelI = subProb_[*cliqI].getNodeLabel();
          int sizCliqI = subProb_[*cliqI].sizCliq_; //OPTIMIZE - used only once
          int cliqOffsetI = subProb_[*cliqI].getCliqOffset(); //OPTIMIZE - used only once
          std::vector<int> nodeOffsetI = subProb_[*cliqI].getNodeOffset(); //OPTIMIZE - used only once

          for (int elemI = 0; elemI != sizCliqI; ++elemI) {
           for (int labelI = 0; labelI != nLabelI[elemI]; ++labelI) {

            int curI = cliqOffsetI + nodeOffsetI[elemI] + labelI;
            dumpHessian<<curI<<" "<<curJ<<" "<<eigenHess_.coeffRef(curI,curJ)<<std::endl;
           } //labelI
          } //elemI
         }
        } //for iterating through cliqNeigh
       } // for labelJ
      } //for elemJ
     } //for cliqJ

     dumpHessian.close();
     dumpb.close();
    }
   }

   std::cout<<"solveCG: exit on residual norm "<<residualNorm<<". Returned step norm "<<iterStep.norm()<<". step l-infinity norm "<<iterStep.maxCoeff()<<". Iteration "<<cntIter<<std::endl;
   exitCond = true;
  }
  else {
   ++cntIter;

   Eigen::VectorXd hessDirProd = getHessVecProd(direc);

   alpha = resDotPreRes/(direc.transpose()*hessDirProd);
   iterStep += alpha*direc;

   Eigen::VectorXd nxtRes;
   if (cntIter % 50 == 0) {
    nxtRes = getHessVecProd(iterStep) - b;
   }
   else {
    nxtRes = residual + alpha*hessDirProd;
   }

   double tPrecond = myUtils::getTime();

   switch (precondFlag_)
   {
    case 0:
     preconRes = diagPrecond(nxtRes);
     break;
    case 1:
     preconRes = blkJacobPrecond(nxtRes);
     break;
    case 2:
     preconRes = applyInCholPrecond(nxtRes);
     break;
    case 3:
     for (std::size_t i = 0; i != nDualVar_; ++i) {
      resVec_[i] = nxtRes[i];
      sVec_[i] = direc[i];
      yVec_[i] = hessDirProd[i];
     }
     //precnd_(&numVarQN_, &numPairQN_, &iopQN_, &newtonIter, &cntIter, sVec_, yVec_, resVec_, prVec_, workVecQN_, &lwQN_, iWorkVecQN_, &liwQN_, &build, &info, &mssg);
     for (std::size_t i = 0; i != nDualVar_; ++i) {
      preconRes[i] = prVec_[i];
     }
     break;
    default:
     preconRes = nxtRes;
     break;
   }

   totPrecondTime += myUtils::getTime() - tPrecond;

   double nxtResDotPreRes = nxtRes.dot(preconRes);
   double beta = nxtResDotPreRes/resDotPreRes;

   direc = -1.0*preconRes + beta*direc;

   resDotPreRes = nxtResDotPreRes;

   residual = nxtRes;
  }

 } //CG while loop

 if ((precondFlag_ == 3) && (newtonIter == 1)) {
  std::ofstream debugWorkVec("debugWorkVec.txt");
  std::ofstream debugIWorkVec("debugIWorkVec.txt");

  for (int i = 0; i != lwQN_; ++i) {
   debugWorkVec<<workVecQN_[i]<<" ";
  }

  debugWorkVec.close();

  for (int i = 0; i != liwQN_; ++i) {
   debugIWorkVec<<iWorkVecQN_[i]<<" ";
  }

  debugIWorkVec.close();
 }

// std::cout<<"preconditioning the residual on the average takes "<<totPrecondTime/cntIter<<std::endl;

 nNonZeroLower_ = 0;
// nzHessLower_.clear();
// hessRowInd_.clear();

 return 0;
}

Eigen::VectorXd dualSys::diagPrecond(Eigen::VectorXd residual) {
 Eigen::VectorXd precondRes(residual.size());

 for (std::size_t i = 0; i != nDualVar_; ++i) {
  precondRes(i) = (1/eigenHess_.coeffRef(i,i))*residual(i);
 }

 return precondRes;
}

Eigen::VectorXd dualSys::blkJacobPrecond(Eigen::VectorXd residual) {
 Eigen::VectorXd precondRes(residual.size());

 int blkOffset = 0;

 for (int cliqCnt = 0; cliqCnt != nCliq_; ++cliqCnt) {
  int blkSiz = subProb_[cliqCnt].getDualVar().size();

  Eigen::VectorXd blkRes(blkSiz);

  for (int i = 0; i != blkSiz; ++i) {
   blkRes(i) = residual(blkOffset + i);
  }

  Eigen::MatrixXd curBlk = blkJacob_[cliqCnt];

  blkRes = curBlk*blkRes;

  for (int i = 0; i != blkSiz; ++i) {
   precondRes(blkOffset + i) = blkRes(i);
  }

  blkOffset += blkSiz;
 }

 return precondRes;
}

int dualSys::compInCholPrecond(int p) {
 double alpha = 0.015625; //an initial lower limit

 int orderMat = nDualVar_;
// int nzInChol = nNonZeroLower_ + orderMat*p;

 int* iwa = new int[3*orderMat];
 double* wa1 = new double[orderMat];
 double* wa2 = new double[orderMat];

 double* nzHessLower = new double[nNonZeroLower_];
 int* hessRowInd = new int[nNonZeroLower_];

 for (int i = 0; i != nNonZeroLower_; ++i) {
  nzHessLower[i] = nzHessLower_[i];
  hessRowInd[i] = hessRowInd_[i];
 }

 dicfs_(&orderMat, &nNonZeroLower_, nzHessLower, hessDiag_, hessColPtr_, hessRowInd, inCholLow_, inCholDiag_, \
         inCholColPtr_, inCholRowInd_, &p, &alpha, iwa, wa1, wa2);

#if 0
 std::cout<<"Lin and More: inc. Cholesky strict lower"<<std::endl;
 for (std::size_t i = 0; i != nNonZeroLower_; ++i) {
  std::cout<<inCholLow_[i]<<" "<<std::endl;
 }

 std::cout<<"Lin and More: inc. Cholesky diagonal"<<std::endl;
 for (std::size_t i = 0; i != nDualVar_; ++i) {
  std::cout<<inCholDiag_[i]<<" "<<std::endl;
 }
#endif
 //exit(0);

 delete[] nzHessLower;
 delete[] hessRowInd;
 delete[] iwa;
 delete[] wa1;
 delete[] wa2;

 return 0;
}

Eigen::VectorXd dualSys::applyInCholPrecond(Eigen::VectorXd residual) {
 char inTask = 'N';
 char outTask = 'T';
 int orderMat = nDualVar_;

 double* r = new double[orderMat];

 for (int i = 0; i != orderMat; ++i) {
  r[i] = residual[i];
 }

 Eigen::VectorXd precondRes(orderMat);

 dstrsol_(&orderMat,inCholLow_,inCholDiag_,inCholColPtr_,inCholRowInd_,r,&inTask);
 dstrsol_(&orderMat,inCholLow_,inCholDiag_,inCholColPtr_,inCholRowInd_,r,&outTask);

 for (int i = 0; i != orderMat; ++i) {
  precondRes[i] = r[i];
 }

 delete[] r;

 return precondRes;
}

int dualSys::setFracAsFeas()
{
 for (int iCliq = 0; iCliq != nCliq_; ++iCliq) {
  subProb_[iCliq].primalCliqConsist_ = subProb_[iCliq].primalCliqFrac_;
 }

 primalConsist_ = primalFrac_;

 return 0;
}

int dualSys::solveFista() {
 double gradBasedDamp = 0;
 bool gradDampFlag = true;

 double L, L0;

 L = L0 = 1;

 double beta = 2;
 double t_k, t_prev;

 t_k = t_prev = 1;

// std::vector<double> x_k(nDualVar_,0);
 std::vector<double> dual_prev = dualVar_;
 std::vector<double> diffVec(nDualVar_);

 momentum_ = dualVar_;

 distributeDualVars();
 distributeMomentum();

 bool contIter = true;

 int cntIter, cntIterTau;

 cntIter = cntIterTau = 0;

 double totEnergyTime = 0;

 //double prevDual = 0;

 double bestPrimalEnergy = 0;

 timeStart_ = myUtils::getTime();

 while (contIter) {
  double tFull = myUtils::getTime();

  ++cntIter;
  ++cntIterTau;

  popGradEnergyFista();

  std::cout<<"solveFista: populating gradient and energy takes "<<myUtils::getTime()-tFull<<std::endl;
  std::cout<<std::flush;

  double gradNorm = myUtils::norm<double>(gradient_, nDualVar_);

  if (gradDampFlag) {
   gradBasedDamp = gradNorm/gradDampFactor_; //earlier, it was 3 ####
   gradDampFlag = false;
  }

  if (gradBasedDamp < dampTol_) {
   gradBasedDamp = dampTol_;
  }

  if ((annealIval_ != -1) && (gradNorm < gradBasedDamp) && (tau_ < tauMax_)) {
   std::cout<<"Annealing!"<<std::endl;
   std::cout<<std::flush;

   tau_ *= tauScale_;
   stgCvxCoeff_ /= tauScale_;

   gradDampFlag = true;
   L = L0; //resetting L ####
  }
  else {
#if 0
  if (tau_ < 64) {
   L = 10;
  } else if (tau_ < 1024) {
   L = 10*pow(beta,10);
  } else if (tau_ < 4096) {
   L = 10*pow(beta,20);
  } else if (tau_ == 4096) {
   L = 10*pow(beta,30);
  } else if (tau_ == 8192) {
   L = 10*pow(beta,40);
  }
#endif

   int gradMaxInd = myUtils::argmaxAbs<double>(gradient_,0,nDualVar_);

   double gradMax = std::abs(gradient_[gradMaxInd]);

   //double dualDiff = std::abs(prevDual-curEnergy_);

   if (((gradMax < gradTol_) || (cntIter > maxIter_) || (pdGap_ < gradTol_) || (smallGapIterCnt_ > 10*cntExitCond_)) && ((-1 == annealIval_) || (tau_ == tauMax_))) {
    contIter = false;
    std::cout<<"solveFista: Exit at tau = "<<tau_<<" gradient l-infinity "<<gradMax<<" Iteration count "<<cntIter<<std::endl;
   }

   double invL = -1./L;

   myUtils::scaleVector<double>(invL,gradient_,dualVar_,nDualVar_);

  //std::cout<<"norm should be 0: "<<myUtils::norm(x_k,nDualVar_)<<std::endl;

  //std::cout<<"energy at y: "<<compEnergy(dualVar_)<<" energy at x_k "<<compEnergy(x_k)<<" energy computed by routine "<<curEnergy_<<std::endl;

   myUtils::addArray<double>(dualVar_,momentum_,dualVar_,nDualVar_);

   myUtils::scaleVector<double>(-1,momentum_,diffVec,nDualVar_);
   myUtils::addArray<double>(dualVar_,diffVec,diffVec,nDualVar_);

   double iterEnergy;

   iterEnergy = (sparseFlag_) ? compEnergySparse(dualVar_):compEnergy(dualVar_);

   int lsCnt = 0;

   double rhsEnergy = curEnergy_ + myUtils::dotVec<double>(gradient_,diffVec,nDualVar_) + (L/2)*myUtils::dotVec<double>(diffVec,diffVec,nDualVar_);

   double backTol = pow(10,-6);

#if 1
   while (iterEnergy > rhsEnergy + backTol) {
    ++lsCnt;
//   std::cout<<"ls count: "<<lsCnt<<" lhs energy: "<<iterEnergy<<" rhs energy: "<<rhsEnergy<<" L: "<<L<<std::endl;

    L *= beta;

    invL = -1./L;

    myUtils::scaleVector<double>(invL,gradient_,dualVar_,nDualVar_);
    myUtils::addArray<double>(dualVar_,momentum_,dualVar_,nDualVar_);

    myUtils::scaleVector<double>(-1,momentum_,diffVec,nDualVar_);
    myUtils::addArray<double>(dualVar_,diffVec,diffVec,nDualVar_);

    iterEnergy = (sparseFlag_) ? compEnergySparse(dualVar_):compEnergy(dualVar_);

    rhsEnergy = curEnergy_ + myUtils::dotVec<double>(gradient_,diffVec,nDualVar_) + (L/2.)*myUtils::dotVec<double>(diffVec,diffVec,nDualVar_);

   std::cout<<"solveFista: Iter energy "<<iterEnergy<<" Rhs energy "<<rhsEnergy<<" L "<<L<<std::endl;
   }
#endif

//  std::cout<<"No. of line search iterations "<<lsCnt<<std::endl;

   t_k = (1 + sqrt(1 + 4*pow(t_prev,2)))/2;

   myUtils::scaleVector<double>(-1,dual_prev,dual_prev,nDualVar_);
   myUtils::addArray<double>(dualVar_,dual_prev,dual_prev,nDualVar_);
   myUtils::scaleVector<double>((t_prev-1)/t_k,dual_prev,dual_prev,nDualVar_);

   myUtils::addArray<double>(dualVar_,dual_prev,momentum_,nDualVar_);

#if 0
  std::cout<<"Step value "<<t_k<<std::endl;

  std::cout<<"Momentum";

  for (std::vector<double>::iterator iMom = momentum_.begin(); iMom != momentum_.end(); ++iMom) {
   std::cout<<" "<<*iMom;
  }

  std::cout<<std::endl;

  std::cout<<"Dual variables";

  for (std::vector<double>::iterator iDual = dualVar_.begin(); iDual != dualVar_.end(); ++iDual) {
   std::cout<<" "<<*iDual;
  }

  std::cout<<std::endl;
#endif

   distributeDualVars();
   distributeMomentum();

   dual_prev = dualVar_;

   t_prev = t_k;

   int cntInterval = 10;

   if (((annealIval_ == -1) || (tau_ == tauMax_)) && (cntIter % cntExitCond_ == 0)) {
    recoverFracPrimal();
    recoverFeasPrimal();

    double curNonSmoothPrimalEnergy = compNonSmoothPrimalEnergy();
    double curNonSmoothDualEnergy = compNonSmoothDualEnergy();

    std::cout<<"solveFista: before update: curNonSmoothDualEnergy "<<curNonSmoothDualEnergy<<" curNonSmoothPrimalEnergy "<<curNonSmoothPrimalEnergy<<" bestNonSmoothPrimalEnergy "<<bestNonSmoothPrimalEnergy_<<std::endl;

    if (pdInitFlag_) {
     bestNonSmoothPrimalEnergy_ = curNonSmoothPrimalEnergy;
     pdInitFlag_ = false;
    }
    else if (curNonSmoothPrimalEnergy > bestNonSmoothPrimalEnergy_) {
     bestNonSmoothPrimalEnergy_ = curNonSmoothPrimalEnergy;
    }

    pdGap_ = (curNonSmoothDualEnergy - bestNonSmoothPrimalEnergy_)/(std::abs(curNonSmoothDualEnergy) + std::abs(bestNonSmoothPrimalEnergy_));

    std::cout<<"solveFista: after update: curNonSmoothDualEnergy "<<curNonSmoothDualEnergy<<" curNonSmoothPrimalEnergy "<<curNonSmoothPrimalEnergy<<" bestNonSmoothPrimalEnergy "<<bestNonSmoothPrimalEnergy_<<std::endl;
    std::cout<<"solveFista: PD Gap "<<pdGap_<<std::endl;

    if (pdGap_ < 10*gradTol_) {
     ++smallGapIterCnt_;
    }
   }

   if (cntIter % cntInterval == 0) {
    std::cout<<"ITERATION "<<cntIter<<" took "<<myUtils::getTime() - tFull<<" seconds. Gradient norm "<<gradNorm<<" Gradient max "<<gradMax<<" Gradient based threshold "<<gradBasedDamp<<" Smoothing "<<tau_<<" L "<<L<<std::endl;
    std::cout<<"strong convexity coefficient "<<stgCvxCoeff_<<std::endl;

    double tEnergy = myUtils::getTime();

    recoverNodeFracPrimal();
    recoverMaxPrimal(primalFrac_);

    double curIntPrimalEnergy;

    curIntPrimalEnergy = compIntPrimalEnergy();

    if (bestPrimalEnergy < curIntPrimalEnergy) {
     bestPrimalEnergy = curIntPrimalEnergy;
     bestPrimalMax_ = primalMax_;
     timeToBestPrimal_ = myUtils::getTime() - timeStart_;
    }
    else if (cntIter == cntInterval) {
     bestPrimalEnergy = curIntPrimalEnergy;
     bestPrimalMax_ = primalMax_;
     timeToBestPrimal_ = myUtils::getTime() - timeStart_;
    }

    std::cout<<std::fixed;
    std::cout<<std::setprecision(6);

    std::cout<<" Smooth dual energy "<<curEnergy_<<std::endl;

    std::cout<<" Best integral primal energy "<<bestPrimalEnergy<<" Current integral primal energy (from feasible) "<<curIntPrimalEnergy;

    std::cout<<" Non-smooth dual energy ";

    if (sparseFlag_) {
     std::cout<<"No efficient implementation yet!"<<std::endl;
    }
    else {
     std::cout<<compNonSmoothDualEnergy()<<std::endl;
    }

    totEnergyTime += myUtils::getTime() - tEnergy;
    //double curEnergyTime = myUtils::getTime() - tEnergy;
   } //if (cntIter % cntInterval == 0)

   //prevDual = curEnergy_;
  } //if ((annealIval_ != -1) && (gradNorm < gradBasedDamp) && (tau_ < tauMax_))
//  std::cout<<"ITERATION "<<cntIter<<" took "<<(myUtils::getTime() - tFull - curEnergyTime)<<" seconds. ITERATION "<<cntIter+1<<" starts, with tau "<<tau_<<"."<<std::endl; //" "<<compNonSmoothDualEnergy()<<" "<<compIntPrimalEnergy()<<std::endl;
 } //MAIN FISTA WHILE LOOP

 std::cout<<"solveFista: total time taken "<<myUtils::getTime() - timeStart_<<" seconds."<<std::endl;

 //recoverFracPrimal();
 //recoverFeasPrimal();
 recoverNodeFracPrimal();
 recoverMaxPrimal(primalFrac_);

 std::cout<<std::fixed;
 std::cout<<std::setprecision(6);

 //std::cout<<"solveFista: Fractional primal energy: "<<compSmoothPrimalEnergy()<<std::endl;
 std::cout<<"solveFista: Smooth dual energy: ";
 if (sparseFlag_) {
  std::cout<<compEnergySparse(dualVar_)<<std::endl;
 }
 else {
  std::cout<<compEnergy(dualVar_)<<std::endl;
 }
 std::cout<<"solveFista: Integral primal energy (feasible primal): "<<compIntPrimalEnergy()<<std::endl;
 recoverMaxPrimal(primalFrac_);
 std::cout<<"solveFista: Best integral primal energy: "<<bestPrimalEnergy<<std::endl;
 std::cout<<"solveFista: Non-smooth dual energy: "<<compNonSmoothDualEnergy()<<std::endl;
 recoverFracPrimal();
 recoverFeasPrimal();
 std::cout<<"solveFista: Non-smooth primal energy:"<<compNonSmoothPrimalEnergy()<<std::endl;
 std::cout<<"Total time spent on computing energies: "<<totEnergyTime<<std::endl;

 return 0;
}

int dualSys::solveSCD()
{
 std::cout<<"solveSCD: Sparse flag "<<sparseFlag_<<std::endl;

 double tCompEnergies = 0;

 bool contIter = true;

 int cntIter = 0, cntIterTau = 0;

 double gradBasedDamp = 0;

 double bestPrimalEnergy = 0;

 timeStart_ = myUtils::getTime();

 while ((contIter) && (cntIter < maxIter_)) {
  double tFull = myUtils::getTime();

  ++cntIter;
  ++cntIterTau;

  //updateMSD(cntIter);
  updateStar(cntIter);

  popGradEnergy();

  double gradNorm = myUtils::norm<double>(gradient_, nDualVar_);

  if (cntIter == 1) {
   gradBasedDamp = gradNorm/gradDampFactor_; //earlier, it was 3 ####
  }

  if ((annealIval_ != -1) && (gradNorm < gradBasedDamp) && (tau_ < tauMax_)) {
   std::cout<<"Annealing: grad norm "<<gradNorm<<" grad based damp "<<gradBasedDamp<<std::endl;
   cntIterTau = 1;
   tau_ *= tauScale_;
   stgCvxCoeff_ /= tauScale_;
   //gradDampFlag = true;

   //updateMSD(cntIter);
   updateStar(cntIter);

   popGradEnergy();

   gradNorm = myUtils::norm<double>(gradient_, nDualVar_);
   gradBasedDamp = gradNorm/gradDampFactor_;

   if (gradBasedDamp < dampTol_) {
    gradBasedDamp = dampTol_;
   }
  }

  int gradMaxInd = myUtils::argmaxAbs<double>(gradient_,0,nDualVar_);

  double gradMax = std::abs(gradient_[gradMaxInd]);

  int checkIter = -1;

  std::string opName;

  if (cntIter == checkIter) {
   opName = "gradient_" + std::to_string(checkIter) + ".txt";

   std::ofstream opGrad(opName);
   opGrad<<std::scientific;
   opGrad<<std::setprecision(6);
   for (std::vector<double>::iterator gradIter = gradient_.begin(); gradIter != gradient_.end(); ++gradIter) {
    opGrad<<*gradIter<<std::endl;
   }
   opGrad.close();

   opName = "dual_" + std::to_string(checkIter) + ".txt";

   std::ofstream opDual(opName);
   opDual<<std::scientific;
   opDual<<std::setprecision(6);
   for (std::vector<double>::iterator dualIter = dualVar_.begin(); dualIter != dualVar_.end(); ++dualIter) {
    opDual<<*dualIter<<std::endl;
   }
   opDual.close();
  }

  if ((gradMax < gradTol_) || (gradNorm < gradTol_) || (pdGap_ < gradTol_) || (smallGapIterCnt_ > 10*cntExitCond_)) {
   if ((annealIval_ == -1) || (tau_ == tauMax_)) {
    contIter = false;

    gradNorm = myUtils::norm<double>(gradient_, nDualVar_);
    gradMaxInd = myUtils::argmaxAbs<double>(gradient_,0,nDualVar_);
    gradMax = std::abs(gradient_[gradMaxInd]);
   }
   else  {
    gradBasedDamp = 2*gradNorm;

    if (gradBasedDamp < 0.000001) {
     gradBasedDamp = 0.000001;
    }
   }
  }

  int cntInterval = 10;

  if (((annealIval_ == -1) || (tau_ == tauMax_)) && (cntIter % cntExitCond_ == 0)) {
   recoverFracPrimal();
   recoverFeasPrimal();

   double curNonSmoothPrimalEnergy = compNonSmoothPrimalEnergy();
   double curNonSmoothDualEnergy = compNonSmoothDualEnergy();

   std::cout<<"solveSCD: before update: curNonSmoothDualEnergy "<<curNonSmoothDualEnergy<<" curNonSmoothPrimalEnergy "<<curNonSmoothPrimalEnergy<<" bestNonSmoothPrimalEnergy "<<bestNonSmoothPrimalEnergy_<<std::endl;

   if (pdInitFlag_) {
    bestNonSmoothPrimalEnergy_ = curNonSmoothPrimalEnergy;
    pdInitFlag_ = false;
   }
   else if (curNonSmoothPrimalEnergy > bestNonSmoothPrimalEnergy_) {
    bestNonSmoothPrimalEnergy_ = curNonSmoothPrimalEnergy;
   }

   pdGap_ = (curNonSmoothDualEnergy - bestNonSmoothPrimalEnergy_)/(std::abs(curNonSmoothDualEnergy) + std::abs(bestNonSmoothPrimalEnergy_));

   std::cout<<"solveSCD: after update: curNonSmoothDualEnergy "<<curNonSmoothDualEnergy<<" curNonSmoothPrimalEnergy "<<curNonSmoothPrimalEnergy<<" bestNonSmoothPrimalEnergy "<<bestNonSmoothPrimalEnergy_<<std::endl;
   std::cout<<"solveSCD: PD Gap "<<pdGap_<<std::endl;

   if (pdGap_ < 10*gradTol_) {
    ++smallGapIterCnt_;
   }
  }

#if 1
  if (cntIter % cntInterval == 0) {
   std::cout<<"solveSCD: ITERATION "<<cntIter<<". Tau "<<tau_<<" and gradient threshold "<<gradBasedDamp<<".";
   std::cout<<" Current iteration took "<<(myUtils::getTime() - tFull)<<" seconds.";
   std::cout<<" Gradient: l-infinity: "<<gradMax<<", Euclidean: "<<gradNorm<<". Energy: "<<curEnergy_;
   std::cout<<"Strong convexity coefficient "<<stgCvxCoeff_;
   std::cout<<std::endl;

   double tEnergy = myUtils::getTime();

   std::cout<<std::fixed;
   std::cout<<std::setprecision(6);

   recoverNodeFracPrimal();
   recoverMaxPrimal(primalFrac_);

   double curIntPrimalEnergy;

   curIntPrimalEnergy = compIntPrimalEnergy();

   if (bestPrimalEnergy < curIntPrimalEnergy) {
    bestPrimalEnergy = curIntPrimalEnergy;
    bestPrimalMax_ = primalMax_;
    timeToBestPrimal_ = myUtils::getTime() - timeStart_;
    std::cout<<"Best primal updated in iteration "<<cntIter<<" at time "<<timeToBestPrimal_<<std::endl;
   }
   else if (cntIter == cntInterval) {
    bestPrimalEnergy = curIntPrimalEnergy;
    bestPrimalMax_ = primalMax_;
    timeToBestPrimal_ = myUtils::getTime() - timeStart_;
   }

   std::cout<<" Best integral primal energy "<<bestPrimalEnergy;

   std::cout<<" Non-smooth dual energy ";

   if (sparseFlag_) {
    std::cout<<"Not efficiently computable yet!";
   }
   else {
    std::cout<<compNonSmoothDualEnergy();
   }

   std::cout<<std::endl;

   std::cout<<"Computing energies took "<<myUtils::getTime() - tEnergy<<" seconds."<<std::endl;

   tCompEnergies += myUtils::getTime() - tEnergy;
  }

#endif
 } //while ((contIter) && (cntIter < maxIter_))

 std::cout<<"solveSCD: total time taken "<<myUtils::getTime() - timeStart_<<" seconds."<<std::endl;

 recoverNodeFracPrimal();
 recoverMaxPrimal(primalFrac_);

 std::cout<<std::fixed;
 std::cout<<std::setprecision(6);

 std::cout<<"solveSCD: Smooth dual energy: ";
 if (sparseFlag_) {
  std::cout<<compEnergySparse(dualVar_)<<std::endl;
 }
 else {
  std::cout<<compEnergy(dualVar_)<<std::endl;
 }
 std::cout<<"solveSCD: Integral primal energy: "<<compIntPrimalEnergy()<<std::endl;

 recoverNodeFracPrimal();
 recoverMaxPrimal(primalFrac_);

 std::cout<<"solveSCD: Best integral primal energy: "<<bestPrimalEnergy<<std::endl;
 std::cout<<"solveSCD: Non-smooth dual energy: "<<compNonSmoothDualEnergy()<<std::endl;
 recoverFracPrimal();
 recoverFeasPrimal();
 std::cout<<"solveSCD: Non-smooth primal energy: "<<compNonSmoothPrimalEnergy()<<std::endl;
 std::cout<<"Total time spent on computing energies in each iteration: "<<tCompEnergies<<std::endl;

 return 0;
} //solveSCD()

int dualSys::updateMSD(int cntIter) {

 std::list<int> unprocessCliq(nCliq_);
 std::vector<std::list<int> > unprocessMemNode(nCliq_);

 int cliqCnt = 0;

 for (std::list<int>::iterator iCliq = unprocessCliq.begin(); iCliq != unprocessCliq.end(); ++iCliq) {
  *iCliq = cliqCnt;

  int sizCliq = subProb_[*iCliq].sizCliq_;

  unprocessMemNode[*iCliq].resize(sizCliq);

  int nodeCnt = 0;

  for (std::list<int>::iterator iNode = unprocessMemNode[*iCliq].begin(); iNode != unprocessMemNode[*iCliq].end(); ++iNode) {
   *iNode = nodeCnt;
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
   int cliqOffset = subProb_[*iCliq].getCliqOffset();
   std::vector<int> memNode = subProb_[*iCliq].memNode_;
   std::vector<int> nodeOffset = subProb_[*iCliq].getNodeOffset();

   int nCliqLab = subProb_[*iCliq].nCliqLab_;
   int sizCliq = subProb_[*iCliq].sizCliq_;
   std::vector<short> nodeLabels = subProb_[*iCliq].getNodeLabel();

   std::vector<double> dualVar = subProb_[*iCliq].getDualVar();

   int subDualSiz = dualVar.size();

   std::vector<double> nodeLabSum(subDualSiz);

   std::uniform_int_distribution<> disNode(1, unprocessMemNode[*iCliq].size());

   std::list<int>::iterator iNode = unprocessMemNode[*iCliq].begin();

   int randNodeInd = disNode(gen) - 1;

   std::advance(iNode, randNodeInd);

   int curNode = memNode[*iNode];

   double expVal;

   int nLabel = nLabel_[curNode];

   double f2DenExpMax;
   std::vector<double> f2LabelSum(nLabel);
   std::vector<double> f2DenExp(nLabel);

   double f2Den = 0;

//  #pragma omp parallel for
   for (int i = 0; i != nLabel; ++i) {
    double cliqSumLabel = 0;

//   #pragma omp for reduction(+ : cliqSumLabel)
    for (std::vector<int>::iterator j = cliqPerNode_[curNode].begin(); j != cliqPerNode_[curNode].end(); ++j) {
     int nodeInd = myUtils::findNodeIndex(subProb_[*j].memNode_, curNode);

     std::vector<double> dualVar = subProb_[*j].getDualVar();

     cliqSumLabel += dualVar[subProb_[*j].getNodeOffset()[nodeInd] + i];
    }
    f2DenExp[i] = uEnergy_[unaryOffset_[curNode] + i] + cliqSumLabel;
   } //for i = [0:nLabel)

   f2DenExpMax = *std::max_element(f2DenExp.begin(), f2DenExp.end());

   for (int i = 0; i != nLabel; ++i) {
    expVal = exp(tau_*(f2DenExp[i] - f2DenExpMax));
    myUtils::checkRangeError(expVal);

    f2LabelSum[i] = expVal;

    f2Den += expVal;
   }

   for (int i = 0; i != nLabel; ++i) {
    f2LabelSum[i] /= f2Den;
   }

   if (!sparseFlag_) {
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

     f1DenExp[i] = subProb_[*iCliq].getCE(i) - nodeSum;

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
     expVal = exp(tau_*(f1DenExp[i] - f1DenExpMax));
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

    for (std::vector<double>::iterator nodeLabIter = nodeLabSum.begin(); nodeLabIter != nodeLabSum.end(); ++nodeLabIter) {
     *nodeLabIter /= f1Den;
    }
   }
   else {
    cliqSPSparseSCD(&subProb_[*iCliq], tau_, *iNode, nodeLabSum);
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
    if (logArg < minNumAllow_) {
     logArg = minNumAllow_;
    }

    logVal = log(logArg);
    myUtils::checkLogError(logVal);

    dualVar_[updateInd] += (1/(2*tau_))*logVal;

    logArg = f2LabelSum[iLabel];
    if (logArg < minNumAllow_) {
     logArg = minNumAllow_;
    }

    logVal = log(logArg);
    myUtils::checkLogError(logVal);

    dualVar_[updateInd] -= (1/(2*tau_))*logVal;
   }

   unprocessMemNode[*iCliq].erase(iNode);

   distributeDualVars();
  }
 }

 return 0;
}

int dualSys::updateStar(int cntIter) {

 std::list<int> unprocessNode(nNode_);

 std::iota(unprocessNode.begin(),unprocessNode.end(),0);

 while (unprocessNode.size() > 0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> disNode(1, unprocessNode.size());

  std::list<int>::iterator iNode = unprocessNode.begin();

  int randNodeInd = disNode(gen) - 1;

  std::advance(iNode, randNodeInd);

  int curNode = *iNode;

  int nLabel = nLabel_[curNode];

  double expVal;

  double f2DenExpMax;
  std::vector<double> f2LabelSum(nLabel);
  std::vector<double> f2DenExp(nLabel);

  double f2Den = 0;

//  #pragma omp parallel for
  for (int i = 0; i != nLabel; ++i) {
   double cliqSumLabel = 0;

//   #pragma omp for reduction(+ : cliqSumLabel)
   for (std::vector<int>::iterator j = cliqPerNode_[curNode].begin(); j != cliqPerNode_[curNode].end(); ++j) {
    int nodeInd = myUtils::findNodeIndex(subProb_[*j].memNode_, curNode);

    std::vector<double> dualVar = subProb_[*j].getDualVar();

    cliqSumLabel += dualVar[subProb_[*j].getNodeOffset()[nodeInd] + i];
   }
   f2DenExp[i] = uEnergy_[unaryOffset_[curNode] + i] + cliqSumLabel;
  } //for i = [0:nLabel)

  f2DenExpMax = *std::max_element(f2DenExp.begin(), f2DenExp.end());

  for (int i = 0; i != nLabel; ++i) {
   expVal = exp(tau_*(f2DenExp[i] - f2DenExpMax));
   myUtils::checkRangeError(expVal);

   f2LabelSum[i] = expVal;

   f2Den += expVal;
  }

  for (int i = 0; i != nLabel; ++i) {
   f2LabelSum[i] /= f2Den;
  }

  std::vector<std::vector<double> > nodeLabSumSet(cliqPerNode_[curNode].size());

  std::vector<double> nodeLabSumLogProd(nLabel);

  int nCliq = 0;
  double cliqPerNode = static_cast<double>(cliqPerNode_[curNode].size());

  for (std::vector<int>::const_iterator iCliq = cliqPerNode_[curNode].begin(); iCliq != cliqPerNode_[curNode].end(); ++iCliq) {
   std::vector<int> memNode = subProb_[*iCliq].memNode_;
   std::vector<int> nodeOffset = subProb_[*iCliq].getNodeOffset();

   int nodeInd = myUtils::findNodeIndex(memNode, curNode);

   int nCliqLab = subProb_[*iCliq].nCliqLab_;
   int sizCliq = subProb_[*iCliq].sizCliq_;
   std::vector<short> nodeLabels = subProb_[*iCliq].getNodeLabel();

   std::vector<double> dualVar = subProb_[*iCliq].getDualVar();

   int subDualSiz = dualVar.size();

   std::vector<double> nodeLabSum(subDualSiz);

   if (!sparseFlag_) {
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

     f1DenExp[i] = subProb_[*iCliq].getCE(i) - nodeSum;

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
     expVal = exp(tau_*(f1DenExp[i] - f1DenExpMax));
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

    for (std::vector<double>::iterator nodeLabIter = nodeLabSum.begin(); nodeLabIter != nodeLabSum.end(); ++nodeLabIter) {
     *nodeLabIter /= f1Den;
    }
   }
   else {
    cliqSPSparseSCD(&subProb_[*iCliq], tau_, nodeInd, nodeLabSum);
   }

   nodeLabSumSet[nCliq] = nodeLabSum;

   for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
   int margInd = nodeOffset[nodeInd] + iLabel;

    double logArg = nodeLabSum[margInd];

    if (logArg < minNumAllow_) {
     logArg = minNumAllow_;
    }

    double logVal = log(logArg);
    myUtils::checkLogError(logVal);

    nodeLabSumLogProd[iLabel] += logVal;
   }

   ++nCliq;
  } //for iCliq

  nCliq = 0;

  for (std::vector<int>::const_iterator iCliq = cliqPerNode_[curNode].begin(); iCliq != cliqPerNode_[curNode].end(); ++iCliq) {
   int cliqOffset = subProb_[*iCliq].getCliqOffset();
   std::vector<int> memNode = subProb_[*iCliq].memNode_;
   std::vector<int> nodeOffset = subProb_[*iCliq].getNodeOffset();

   int nodeInd = myUtils::findNodeIndex(memNode, curNode);

   for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
    int margInd = nodeOffset[nodeInd] + iLabel;
    int updateInd = cliqOffset + margInd;

    double logArg = f2LabelSum[iLabel];

    if (logArg < minNumAllow_) {
     logArg = minNumAllow_;
    }

    double logVal = log(logArg);
    myUtils::checkLogError(logVal);

    //std::cout<<"updateStar: nodeLabSumLogProd: "<<nodeLabSumLogProd[iLabel]<<" log(f2LabelSum[iLabel]) "<<logVal<<" update value one "<<(-1/(cliqPerNode_[curNode].size()+1))*(1/tau_)*(logVal + nodeLabSumLogProd[iLabel]);

    dualVar_[updateInd] -= (1/(cliqPerNode+1))*(1/tau_)*(logVal + nodeLabSumLogProd[iLabel]);

    logArg = nodeLabSumSet[nCliq][margInd];

    if (logArg < minNumAllow_) {
     logArg = minNumAllow_;
    }

    logVal = log(logArg);
    myUtils::checkLogError(logVal);

    dualVar_[updateInd] += (1/tau_)*logVal;

    //std::cout<<" value two "<<(1/tau_)*logVal<<std::endl;
   }

   ++nCliq;
  }

  distributeDualVars();

  unprocessNode.erase(iNode);
 } //while loop

 return 0;
}

int dualSys::solveTrueLM()
{
 double tCompEnergies = 0;

 bool contIter = true;

 int cntIter = 0, cntIterTau = 1;

 double gradBasedDamp = 0;
 bool gradDampFlag = true;

 dampLambda_ = 0.1; //####
 double prevLambda = dampLambda_; //needed for LM-update when entire Hessian and gradient need not be computed

 bool freshComp = true;

 while ((contIter) && (cntIter < maxIter_)) {
  double tFull = myUtils::getTime();

  ++cntIter;

  eigenHess_.reserve(reserveHess_);

  double tGHE = myUtils::getTime(); //time to compute gradient, hessian and current energy

  if (freshComp) {
   popGradHessEnergy(cntIter);
  }
  else {
   for (std::size_t i = 0 ; i != nDualVar_; ++i) {
    eigenHess_.coeffRef(i,i) += dampLambda_ - prevLambda;
   }
  }

  std::cout<<"solveNewton: populated gradient, hessian and energy "<<(myUtils::getTime() - tGHE)<<" seconds."<<std::endl;

  double gradNorm = myUtils::norm<double>(gradient_, nDualVar_);

  if (gradDampFlag) {
   gradBasedDamp = gradNorm/gradDampFactor_;
   gradDampFlag = false;
  }

  if ((annealIval_ != -1) && (gradNorm < gradBasedDamp) && (tau_ < tauMax_)) {
   tau_ *= tauScale_;
   gradDampFlag = true; //earlier, it was 3 ####
   dampLambda_ = 0.1;
  }

  int gradMaxInd = myUtils::argmaxAbs<double>(gradient_,0,nDualVar_);

  double gradMax = std::abs(gradient_[gradMaxInd]);

  std::cout<<"solveNewton: gradient l-infinity norm: "<<gradMax<<", Euclidean norm: "<<gradNorm<<". Energy: "<<curEnergy_<<std::endl;

  int checkIter = -1;

  std::string opName;

  if (cntIter == checkIter) {
   opName = "gradient_" + std::to_string(checkIter) + ".txt";

   std::ofstream opGrad(opName);
   opGrad<<std::scientific;
   opGrad<<std::setprecision(6);
   for (std::vector<double>::iterator gradIter = gradient_.begin(); gradIter != gradient_.end(); ++gradIter) {
    opGrad<<*gradIter<<std::endl;
   }
   opGrad.close();
  }

  if (cntIter == checkIter) {
   opName = "dual_" + std::to_string(checkIter) + ".txt";

   std::ofstream opDual(opName);
   opDual<<std::scientific;
   opDual<<std::setprecision(6);
   for (std::vector<double>::iterator dualIter = dualVar_.begin(); dualIter != dualVar_.end(); ++dualIter) {
    opDual<<*dualIter<<std::endl;
   }
   opDual.close();
  }

  if (((tau_ < tauMax_) || (gradMax > gradTol_)) && (gradNorm > gradTol_)) {
   Eigen::VectorXd eigenStep(nDualVar_), eigenGuess(nDualVar_);

   std::ofstream hessFile;

   if (cntIter == checkIter) {
    opName = "hessian_" + std::to_string(checkIter) + ".txt";

    hessFile.open(opName);
    hessFile<<std::scientific;
    hessFile<<std::setprecision(6);

    for (int cJ = 0; cJ != nCliq_; ++cJ) {
     int sizCliqJ = subProb_[cJ].sizCliq_;
     int offsetOne = subProb_[cJ].getCliqOffset();
     std::set<int> cliqNeigh = subProb_[cJ].cliqNeigh_;

     std::vector<int> nodeOffsetJ = subProb_[cJ].getNodeOffset();
     std::vector<short> nLabelJ = subProb_[cJ].getNodeLabel();

     for (int nJ = 0; nJ != sizCliqJ; ++nJ) {
      for (int lJ = 0; lJ != nLabelJ[nJ]; ++lJ) {
       int curJ = offsetOne + nodeOffsetJ[nJ] + lJ;

       for (std::set<int>::iterator cI = cliqNeigh.begin(); cI != cliqNeigh.end(); ++cI) {
        std::vector<int> nodeOffsetI = subProb_[*cI].getNodeOffset();
        std::vector<short> nLabelI = subProb_[*cI].getNodeLabel();

        for (int nI = 0; nI != subProb_[*cI].sizCliq_; ++nI) {
         for (int lI = 0; lI != nLabelI[nI]; ++lI) {
          int curI = subProb_[*cI].getCliqOffset() + nodeOffsetI[nI] + lI;

          hessFile<<curI<<" "<<curJ<<" "<<eigenHess_.coeffRef(curI,curJ)<<std::endl;
         }
        }
       }
      }
     }
    }

    hessFile.close();
   }

   Eigen::VectorXd oriGrad(nDualVar_);

   //custom CG implementation
   for (std::size_t i = 0; i < nDualVar_; ++i) {
    oriGrad[i] = gradient_[i];
    eigenGrad_[i] = gradient_[i];
   }

   double tCG = myUtils::getTime();
   std::vector<Eigen::VectorXd> stepBackVec;
   solveCG(eigenStep, stepBackVec, cntIterTau);
   std::cout<<"CG took "<<myUtils::getTime() - tCG<<" seconds."<<std::endl;

   for (std::size_t i = 0; i != nDualVar_; ++i) {
    newtonStep_[i] = eigenStep[i];
    eigenGuess[i] = 0; //eigenStep[i]; //####
   }

   Eigen::VectorXd eigenInter = eigenHess_.selfadjointView<Eigen::Lower>()*eigenStep;

   double interValOne = eigenStep.dot(eigenInter);

   interValOne *= 0.5;

   double interValTwo = eigenStep.dot(oriGrad);

   double nxtApproxEnergyDiff = interValOne + interValTwo; //diff. wrt current energy

   std::vector<double> z(nDualVar_);

   myUtils::addArray<double>(dualVar_,newtonStep_,z,nDualVar_);

   double nxtEnergy = (sparseFlag_) ? compEnergySparse(z):compEnergy(z);

   double rho = (nxtEnergy - curEnergy_)/nxtApproxEnergyDiff;

   if ((isnan(rho)) || (isinf(rho))) {
    std::cout<<"rho debug: nxtApproxEnergyDiff "<<nxtApproxEnergyDiff<<" nxtEnergy "<<nxtEnergy<<" curEnergy_ "<<curEnergy_<<std::endl;
   }

   prevLambda = dampLambda_;
   double eta_1 = 0.25, eta_2 = 0.5, eta_3 = 0.9, sigma_1 = 2, sigma_2 = 0.5, sigma_3 = 0.25;

   //updating damping lambda inspired by "Newton's method for large-scale optimization" by Bouaricha et al
   if (rho <= eta_1) {
    dampLambda_ *= sigma_1;
   }
   else if ((rho > eta_1) && (rho <= eta_2)) {
    dampLambda_ = dampLambda_;
   }
   else if ((rho > eta_2) && (rho <= eta_3)) {
    dampLambda_ *= sigma_2;
   }
   else if (rho > eta_3) {
    dampLambda_ *= sigma_3;
   }

   std::cout<<"trust region param rho "<<rho<<" updated lambda "<<dampLambda_<<std::endl;

//   std::cout<<"next energy "<<nxtEnergy<<" "<<" current energy "<<curEnergy_<<" next approx energy diff "<<nxtApproxEnergyDiff<<" interValOne "<<interValOne<<" interValTwo "<<interValTwo<<std::endl;

//   if (tau_ == tauMax_) {
    if (rho >= 0.001) {
     myUtils::addArray<double>(dualVar_,newtonStep_,dualVar_,nDualVar_);
     freshComp = true;
     eigenHess_.setZero(); //change logic to insert in first iteration and coeffRef in subsequent ones.
    }
    else {
     freshComp = false;
    }
//   }
#if 0
   else {
    int newtonMaxInd = myUtils::argmaxAbs<double>(newtonStep_,0,nDualVar_);
    double newtonStepNorm = myUtils::norm<double>(newtonStep_, nDualVar_);
//   std::cout<<"solveNewton: before line search newton step l-infinity norm: "<<newtonStep_[newtonMaxInd]<<". Euclidean norm: "<<newtonStepNorm<<std::endl;

    double tLS = myUtils::getTime(); //time to perform line search

    int lsCnt = performLineSearch();
    newtonMaxInd = myUtils::argmaxAbs<double>(newtonStep_,0,nDualVar_);
    newtonStepNorm = myUtils::norm<double>(newtonStep_, nDualVar_);
//   std::cout<<"solveNewton: after line search newton step l-infinity norm: "<<newtonStep_[newtonMaxInd]<<". Euclidean norm: "<<newtonStepNorm<<std::endl;
    std::cout<<"solveNewton: performed line search "<<lsCnt<<" times, "<<(myUtils::getTime() - tLS)<<" seconds."<<std::endl;

    myUtils::addArray<double>(dualVar_,newtonStep_,dualVar_,nDualVar_);
   }
#endif

//   int dualVarInd = myUtils::argmaxAbs<double>(dualVar_,0,nDualVar_);
//   double dualNorm = myUtils::norm<double>(dualVar_, nDualVar_);
//   std::cout<<"solveNewton: dual l-infinity norm: "<<dualVar_[dualVarInd]<<". Euclidean norm: "<<dualNorm<<std::endl;

   distributeDualVars();
  }
  else {
   contIter = false;

   gradNorm = myUtils::norm<double>(gradient_, nDualVar_);
   gradMaxInd = myUtils::argmaxAbs<double>(gradient_,0,nDualVar_);
   gradMax = std::abs(gradient_[gradMaxInd]);

   std::cout<<"solveNewton: gradient l-infinity norm: "<<gradMax<<", Euclidean norm: "<<gradNorm<<". Energy: "<<curEnergy_<<std::endl;
  }

#if 1
  double tEnergy = myUtils::getTime();

  recoverFracPrimal();
  recoverFeasPrimal();
  recoverMaxPrimal(primalConsist_);

  std::cout<<"Fractional primal energy: "<<compSmoothPrimalEnergy()<<" Smooth dual energy: "<<curEnergy_<<" Integral primal energy (feasible): "<<compIntPrimalEnergy();

  recoverMaxPrimal(primalFrac_);
  std::cout<<" Integral primal energy (directly from dual): "<<compIntPrimalEnergy()<<" Non-smooth dual energy: "<<compNonSmoothDualEnergy()<<std::endl;

  std::cout<<"Computing energies took "<<myUtils::getTime() - tEnergy<<" seconds."<<std::endl;
  std::cout<<"NOTE: ITERATION "<<cntIter<<" took "<<(tEnergy - tFull)<<" seconds. The following figure includes unnecessary energy computation."<<std::endl;

  tCompEnergies += myUtils::getTime() - tEnergy;
#endif

  std::cout<<"ITERATION "<<cntIter<<" took "<<(myUtils::getTime() - tFull)<<" seconds. ITERATION "<<cntIter+1<<" starts, with tau "<<tau_<<"."<<std::endl; //" "<<compNonSmoothDualEnergy()<<" "<<compIntPrimalEnergy()<<std::endl;
 }

 recoverFracPrimal();
 recoverFeasPrimal();
 recoverMaxPrimal(primalConsist_);

 std::cout<<"solveNewton: Fractional primal energy: "<<compSmoothPrimalEnergy()<<std::endl;
 std::cout<<"solveNewton: Smooth dual energy: ";

 if (sparseFlag_) {
  std::cout<<compEnergySparse(dualVar_)<<std::endl;
 }
 else {
  std::cout<<compEnergy(dualVar_)<<std::endl;
 }
 std::cout<<"solveNewton: Integral primal energy (feasible): "<<compIntPrimalEnergy()<<std::endl;

 recoverMaxPrimal(primalFrac_);
 std::cout<<"solveNewton: Integral primal energy (directly from dual): "<<compIntPrimalEnergy()<<std::endl;
 std::cout<<"solveNewton: Non-smooth dual energy: "<<compNonSmoothDualEnergy()<<std::endl;

 std::cout<<"Total time spent on computing energies in each iteration: "<<tCompEnergies<<std::endl;

 return 0;
}

Eigen::VectorXd dualSys::getHessVecProd(const Eigen::VectorXd &ipVec) {
 Eigen::VectorXd opVec;

 if (fullHessStoreFlag_) {
  opVec = eigenHess_.selfadjointView<Eigen::Lower>()*ipVec;
 }
 else {
//  double debugTime = myUtils::getTime();
//  opVec = getDistHessVecProd(ipVec);
//  std::cout<<"getHessVecProd: dist multiplication time "<<myUtils::getTime()-debugTime<<std::endl;

//  for (int iDebug = 0; iDebug != nDualVar_; ++iDebug) {
//   std::cout<<" "<<opVec[iDebug];
//  }
//  std::cout<<std::endl;

//  debugTime = myUtils::getTime();
  opVec = getExplicitHessVecProd(ipVec);
//  std::cout<<"getHessVecProd: explicit multiplication time "<<myUtils::getTime()-debugTime<<std::endl;

//  for (int iDebug = 0; iDebug != nDualVar_; ++iDebug) {
//   std::cout<<" "<<opVec[iDebug];
//  }
//  std::cout<<std::endl;
 }

 return opVec;
}

Eigen::VectorXd dualSys::getOptimHessVecProd(const Eigen::VectorXd &ipVec) {
 Eigen::VectorXd opVec(nDualVar_);

 opVec = cliqBlkHess_.selfadjointView<Eigen::Lower>()*ipVec;

 for (int iNode = 0; iNode != nNode_; ++iNode) {
  //Eigen::VectorXd curProd = nodeBlkHess_[iNode].selfadjointView<Eigen::Lower>()*ipVec;
  Eigen::MatrixXd nodeBlk = nodeBlkHess_[iNode];
  Eigen::VectorXd blkVec(nLabCom_);
  blkVec.setZero(nLabCom_);

  for (std::vector<int>::iterator iNodeOff = sparseNodeOffset_[iNode].begin(); iNodeOff != sparseNodeOffset_[iNode].end(); ++iNodeOff) {
   Eigen::VectorXd iterVec(nLabCom_);

   for (int iPos = *iNodeOff; iPos != *iNodeOff + nLabCom_; ++iPos) {
    iterVec[iPos - *iNodeOff] = ipVec[iPos];
   }

   blkVec += nodeBlk.selfadjointView<Eigen::Lower>()*iterVec;
  }

  for (std::vector<int>::iterator iNodeOff = sparseNodeOffset_[iNode].begin(); iNodeOff != sparseNodeOffset_[iNode].end(); ++iNodeOff) {
   for (int iPos = *iNodeOff; iPos != *iNodeOff + nLabCom_; ++iPos) { //current logic assumes all nodes have same number of labels
    opVec[iPos] += blkVec[iPos - *iNodeOff];
   }
  }
 }

 return opVec;
}

Eigen::VectorXd dualSys::getDistHessVecProd(const Eigen::VectorXd &ipVec) {
 Eigen::VectorXd opVec(nDualVar_);

 //opVec = cliqBlkHess_.selfadjointView<Eigen::Lower>()*ipVec;

 int blkOffset = 0;

 for (int iCliq = 0; iCliq != nCliq_; ++iCliq) {
  int blkSiz = subProb_[iCliq].getDualVar().size();

  Eigen::VectorXd blkRes(blkSiz);

  for (int i = 0; i != blkSiz; ++i) {
   blkRes(i) = ipVec(blkOffset + i);
  }

  Eigen::MatrixXd curBlk = blkDiag_[iCliq];

  blkRes = curBlk*blkRes;

  for (int i = 0; i != blkSiz; ++i) {
   opVec(blkOffset + i) = blkRes(i);
  }

  blkOffset += blkSiz;
 } //for iCliq

 for (int iNode = 0; iNode != nNode_; ++iNode) {
  //Eigen::VectorXd curProd = nodeBlkHess_[iNode].selfadjointView<Eigen::Lower>()*ipVec;
  Eigen::MatrixXd nodeBlk = nodeBlkHess_[iNode];
  //Eigen::VectorXd blkVec(nLabCom_);
  //blkVec.setZero(nLabCom_);

  double* distVec = new double[nLabCom_*sparseNodeOffset_[iNode].size()]();

  std::size_t curSet = 0;

  for (std::vector<int>::iterator iNodeOff = sparseNodeOffset_[iNode].begin(); iNodeOff != sparseNodeOffset_[iNode].end(); ++iNodeOff) {
   Eigen::VectorXd iterVec(nLabCom_);

   for (int iPos = *iNodeOff; iPos != *iNodeOff + nLabCom_; ++iPos) {
    iterVec[iPos - *iNodeOff] = ipVec[iPos];
   }

   iterVec = nodeBlk.selfadjointView<Eigen::Lower>()*iterVec;

   for (std::size_t iSet = 0; iSet != sparseNodeOffset_[iNode].size(); ++iSet) {
    if (iSet != curSet) {
     for (int iLabel = 0; iLabel != nLabCom_; ++iLabel) {
      distVec[iSet*nLabCom_ + iLabel] += iterVec[iLabel];
     }
    }
    else {
     for (int iLabel = 0; iLabel != nLabCom_; ++iLabel) {
      distVec[iSet*nLabCom_ + iLabel] += opVec[*iNodeOff + iLabel];
     }
    }
   }

   ++curSet;
  }

#if 1
  curSet = 0;

  for (std::vector<int>::iterator iNodeOff = sparseNodeOffset_[iNode].begin(); iNodeOff != sparseNodeOffset_[iNode].end(); ++iNodeOff) {
   for (int iPos = *iNodeOff; iPos != *iNodeOff + nLabCom_; ++iPos) { //current logic assumes all nodes have same number of labels
    opVec[iPos] = distVec[curSet*nLabCom_ + iPos - *iNodeOff];
   }

   ++curSet;
  }
#endif

  delete[] distVec;
 }

 return opVec;
}

Eigen::VectorXd dualSys::getExplicitHessVecProd(const Eigen::VectorXd &ipVec) {
 Eigen::VectorXd opVec(nDualVar_);

 //std::cout<<"getExplicitHessVecProd"<<std::endl;

// Eigen::MatrixXd debugHess(nDualVar_,nDualVar_);

 for (std::size_t iRow = 0; iRow != nDualVar_; ++iRow) {
//  for (int iCol = 0; iCol != nDualVar_; ++iCol) {
//   debugHess(iRow,iCol) = 0;
//  }

  opVec(iRow) = 0;
 }

 #pragma omp parallel for
 for (int iCliq = 0; iCliq < nCliq_; ++iCliq) {
  std::vector<int> memNode = subProb_[iCliq].getMemNode();
  int subDualSiz = subProb_[iCliq].getDualSiz();
  int cliqOffset = subProb_[iCliq].getCliqOffset();
  std::vector<int> nodeOffset = subProb_[iCliq].getNodeOffset();

  Eigen::MatrixXd cliqBlk = blkDiag_[iCliq];

  int nodeCnt = 0;

  for (std::vector<int>::const_iterator iNode = memNode.begin(); iNode != memNode.end(); ++iNode) {
   Eigen::MatrixXd nodeBlk = nodeBlkHess_[*iNode];
   //double *nodeBlk = nodeBlk_[*iNode];

   int opOffset = cliqOffset + nodeOffset[nodeCnt];

   for (int iLabel = 0; iLabel != nLabCom_; ++iLabel) {
    std::vector<int>::const_iterator iCliqPerNode = cliqPerNode_[*iNode].begin();

    for (std::vector<int>::const_iterator iNodeOff = sparseNodeOffset_[*iNode].begin(); iNodeOff != sparseNodeOffset_[*iNode].end(); ++iNodeOff)  {
     Eigen::VectorXd iterVec(nLabCom_);

     if (*iCliqPerNode != iCliq) {
      for (int jLabel = 0; jLabel != nLabCom_; ++jLabel) { //iLabel:old
       opVec(opOffset + iLabel) += nodeBlk.coeff(iLabel,jLabel)*ipVec.coeff(*iNodeOff + jLabel);
       //opVec(opOffset + iLabel) += nodeBlk[iLabel*nLabCom_+jLabel]*ipVec(*iNodeOff + jLabel);
//       debugHess(opOffset + iLabel,*iNodeOff + jLabel) = nodeBlk(iLabel,jLabel);
//       std::cout<<nodeBlk(iLabel,jLabel)<<" ";
      }

//      for (int jLabel = iLabel; jLabel != nLabCom_; ++jLabel) {
//       opVec(opOffset + iLabel) += nodeBlk(jLabel,iLabel)*ipVec(*iNodeOff + jLabel);
       //opVec(opOffset + iLabel) += nodeBlk[iLabel*nLabCom_+jLabel]*ipVec(*iNodeOff + jLabel);
//       debugHess(opOffset + iLabel,*iNodeOff + jLabel) = nodeBlk(jLabel,iLabel);
//       std::cout<<nodeBlk(jLabel,iLabel)<<" ";
//      }
     }
     else {
      for (int jPos = 0; jPos != subDualSiz; ++jPos) {
       opVec(opOffset + iLabel) += cliqBlk.coeff(jPos,nodeOffset[nodeCnt]+iLabel)*ipVec.coeff(cliqOffset+jPos);
//       debugHess(opOffset + iLabel,cliqOffset+jPos) = cliqBlk(nodeOffset[nodeCnt]+iLabel,jPos);
//       std::cout<<cliqBlk(nodeOffset[nodeCnt]+iLabel,jPos)<<" ";
      }
     }

     //std::cout<<" "<<*iCliqPerNode;

     std::advance(iCliqPerNode,1);
    }//for iNodeOff

//    std::cout<<std::endl;
   }//for iLabel
   //std::cout<<std::endl;

   ++nodeCnt;
  } //for iNode
 } //for iCliq

#if 0
 std::ofstream opHessian("reconHessian.txt");

 for (int iRow = 0; iRow != nDualVar_; ++iRow) {
  for (int iCol = 0; iCol != (nDualVar_-1); ++iCol) {
   opHessian<<debugHess(iRow,iCol)<<" ";
  }
  opHessian<<debugHess(iRow,nDualVar_-1)<<std::endl;
 }

 opHessian.close();

 std::cout<<"Actual explicitHessVecProd:"<<std::endl;

 for (int iDebug = 0; iDebug != nDualVar_; ++iDebug) {
  std::cout<<" "<<opVec[iDebug];
 }
 std::cout<<std::endl;

 opVec = debugHess*ipVec;

 std::cout<<"After populating hessian. explicitHessVecProd:"<<std::endl;

 for (int iDebug = 0; iDebug != nDualVar_; ++iDebug) {
  std::cout<<" "<<opVec[iDebug];
 }
 std::cout<<std::endl;

 opVec = eigenHess_.selfadjointView<Eigen::Lower>()*ipVec;

 std::cout<<"originalHessVecProd"<<std::endl;

 for (int iDebug = 0; iDebug != nDualVar_; ++iDebug) {
  std::cout<<" "<<opVec[iDebug];
 }
 std::cout<<std::endl;

 opVec = getDistHessVecProd(ipVec);

 std::cout<<"DistHessVecProd"<<std::endl;

 for (int iDebug = 0; iDebug != nDualVar_; ++iDebug) {
  std::cout<<" "<<opVec[iDebug];
 }
 std::cout<<std::endl;
#endif

 return opVec;
}

int debugEigenVec(Eigen::VectorXd debugEigen)
{
 int vecSiz = debugEigen.size();

 std::vector<double> debugStl(vecSiz);

 for (int i = 0; i != vecSiz; ++i) {
  debugStl[i] = debugEigen(i);
 }

 return 0;
}
