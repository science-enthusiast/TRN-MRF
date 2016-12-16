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
#include "dualSys.hpp"
#include "myUtils.hpp"
#include "hessVecMult.hpp"
#include "quasiNewton.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/src/IterativeSolvers/IncompleteCholesky.h>
#include <Eigen/src/Core/util/Constants.h>

dualSys::dualSys(int nNode, std::vector<short> nLabel, double tau, int mIter, int annealIval): nDualVar_(0), nNode_(nNode), numLabTot_(0), nLabel_(nLabel), nCliq_(0), tau_(tau), tauStep_(tau), solnQual_(false), finalEnergy_(0), maxIter_(mIter), lsAlpha_(1), annealIval_(annealIval) {

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

 std::vector<short> labelList;

 for (std::vector<int>::const_iterator i = nodeList.begin(); i != nodeList.end(); ++i) {
  cliqPerNode_[*i].push_back(nCliq_);
  labelList.push_back(nLabel_[*i]);
 }

 subProb_.push_back(subProblem(nodeList, labelList, cEnergy));

 cEnergy2DVec_.push_back(*cEnergy); //vector of all pointers to clique energies
 cliqOffset_.push_back(cEnergy2DVec_.size() - 1);

 return ++nCliq_;
}

int dualSys::prepareDualSys() {
 maxCliqSiz_ = 0;
 maxNumNeigh_ = 0;
 nDualVar_ = 0;

 tauMax_ = 8192;

 nLabelMax_ = *std::max_element(nLabel_.begin(), nLabel_.end());

 std::vector<int> reserveVec; //::Constant(nDualVar_,maxNumNeigh_*maxCliqSiz_*nLabelMax_);

 for (int i = 0; i != nCliq_; ++i) {
  subProb_[i].setCliqOffset(nDualVar_);

  if (subProb_[i].sizCliq_ > maxCliqSiz_) {
   maxCliqSiz_ = subProb_[i].sizCliq_;
  }

  nDualVar_ += subProb_[i].getDualVar().size();

  for (std::vector<int>::iterator j = subProb_[i].memNode_.begin(); j != subProb_[i].memNode_.end(); ++j) {
   subProb_[i].cliqNeigh_.insert(cliqPerNode_[*j].begin(), cliqPerNode_[*j].end());
  }
 }

 blkJacob_.resize(nCliq_);

 unsigned long long elemCnt = 0; //####

 for (int i = 0; i != nCliq_; ++i) {
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

  for (std::size_t j = 0; j != subProb_[i].getDualVar().size(); ++j) {
   reserveVec.push_back(reserveSiz);
   elemCnt += reserveSiz;
  }
 }

// std::cout<<"non-zero reserved size for hessian "<<elemCnt<<std::endl;

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

 eigenHess_.resize(nDualVar_,nDualVar_);
 eigenGrad_.resize(nDualVar_);

 newtonStep_.resize(nDualVar_);

 //inc. Cholesky preconditioner
 hessDiag_ = new double[nDualVar_];
 hessColPtr_ = new int[nDualVar_+1];
 nNonZeroLower_ = 0;

// assignPrimalVars("test0000059.uai.txt"); //assign ground truth labeling
// std::cout<<"ground truth energy: "<<compIntPrimalEnergy()<<std::endl;

// recoverFracPrimal();
// recoverMaxPrimal(primalFrac_);

// std::cout<<"Initial integral primal energy: "<<compIntPrimalEnergy()<<std::endl; 
// std::cout<<"Initial non-smooth dual energy: "<<computeNonsmoothDualEnergy()<<std::endl;

 distributeDualVars();

 return 0;
}

int dualSys::popSparseGradHessEnergy() {
 //####
 int totOffDiag = 0, totNegOffDiagH1 = 0, totNegOffDiagH2 = 0;

 nzHessLower_.clear();
 hessRowInd_.clear();

 double f2DenExpMax;
 std::vector<double> f2LabelSum(numLabTot_);

 curEnergy_ = 0;

 unsigned long long elemCnt = 0; //check non-zeros of hessian

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
  }

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
 }

 double f1Den;

#if 1
// #pragma omp parallel for
 for (int cliqJ = 0; cliqJ < nCliq_; ++cliqJ) {
  //std::cout<<"clique "<<cliqJ<<std::endl;
  int nCliqLab = subProb_[cliqJ].nCliqLab_;
  int sizCliqJ = subProb_[cliqJ].sizCliq_;
  std::vector<short> nLabelJ = subProb_[cliqJ].getNodeLabel();

  std::vector<double> dualVar = subProb_[cliqJ].getDualVar();
  std::vector<std::pair<std::vector<int>,double> > sparseCEnergy = subProb_[cliqJ].getSparseCEnergy();
  std::vector<int> nodeOffsetJ = subProb_[cliqJ].getNodeOffset();
  int cliqOffsetJ = subProb_[cliqJ].getCliqOffset();
  std::set<int> cliqNeigh = subProb_[cliqJ].cliqNeigh_;
  std::vector<int> memNodeJ = subProb_[cliqJ].memNode_;

  int subDualSiz = dualVar.size();

  std::vector<double> nodeLabSum(subDualSiz);
  std::vector<double> nodePairLabSum(subDualSiz*subDualSiz);

  Eigen::MatrixXd cliqBlk(subDualSiz,subDualSiz);

  double expVal;

  f1Den = 0;

  std::vector<double> f1DenExp(nCliqLab);

  double f1DenExpMax = 0;

  for (std::vector<std::pair<std::vector<short>,double> > iSparse = sparseCEnergy.begin(); iSparse != sparseCEnergy.end(); ++iSparse) {
   std::vector<short> cliqLab = (*iSparse).first;

   double nodeSum = 0;

   for (int j = 0; j != sizCliqJ; ++j) {
    nodeSum += dualVar[nodeOffsetJ[j] + cliqLab[j]];
   }

   f1DenExp[i] = (*iSparse).second - nodeSum;

   if (i == 0) {
    f1DenExpMax = f1DenExp[i];
   }
   else { 
    if (f1DenExp[i] > f1DenExpMax) {
     f1DenExpMax = f1DenExp[i];
    }
   }
  }

  int iCliqLab = 0;

//  double f1DenExpMax = *std::max_element(f1DenExp.begin(), f1DenExp.end());
  for (std::vector<std::pair<std::vector<short>,double> > iSparse = sparseCEnergy.begin(); iSparse != sparseCEnergy.end(); ++iSparse) {
   expVal = exp(tau_*(f1DenExp[iCliqLab] - f1DenExpMax));
   myUtils::checkRangeError(expVal);
   f1Den += expVal;

//   expVal = exp(tau_*(cEnergy[i] - f1NodeSum[i] - f1DenExpMax));
//   myUtils::checkRangeError(expVal);

   std::vector<short> cliqLab = (*iSparse).first;

   for (int j = 0; j != sizCliqJ; ++j) {
    nodeLabSum[nodeOffsetJ[j] + cliqLab[j]] += expVal;

    for (int k = 0; k != j; ++k) {
     nodePairLabSum[(nodeOffsetJ[k] + cliqLab[k])*subDualSiz + nodeOffsetJ[j] + cliqLab[j]] += expVal;
    }
   }

   ++iCliqLab;
  }

  for (std::vector<double>::iterator nodeLabIter = nodeLabSum.begin(); nodeLabIter != nodeLabSum.end(); ++nodeLabIter) {
   *nodeLabIter /= f1Den;
  }

  for (std::vector<double>::iterator nodePairLabIter = nodePairLabSum.begin(); nodePairLabIter != nodePairLabSum.end(); ++nodePairLabIter) {
   *nodePairLabIter /= f1Den;
  }

  curEnergy_ += (1/tau_)*log(f1Den) + f1DenExpMax;

  int blkCol = 0;

#if 1
  for (int elemJ = 0; elemJ != sizCliqJ; ++elemJ) {
   for (int labelJ = 0; labelJ != nLabelJ[elemJ]; ++labelJ) {
    int curJ = cliqOffsetJ + nodeOffsetJ[elemJ] + labelJ;

    int curJNode = memNodeJ[elemJ];

    gradient_[curJ] = -1*nodeLabSum[nodeOffsetJ[elemJ] + labelJ] + f2LabelSum[unaryOffset_[curJNode] + labelJ];

    if (isnan(gradient_[curJ])) {
     std::cout<<"GRADIENT ENTRY IS NAN!"<<std::endl;
    }

    hessColPtr_[curJ] = nNonZeroLower_+1;
    //hessColPtr_[curJ] = nNonZeroLower_;

    for (std::set<int>::iterator cliqI = cliqNeigh.begin(); cliqI != cliqNeigh.end(); ++cliqI) {
    if (*cliqI >= cliqJ) {
#if 1
     std::vector<short> nLabelI = subProb_[*cliqI].getNodeLabel();
     int sizCliqI = subProb_[*cliqI].sizCliq_; //OPTIMIZE - used only once
     int cliqOffsetI = subProb_[*cliqI].getCliqOffset(); //OPTIMIZE - used only once
     std::vector<int> memNodeI = subProb_[*cliqI].memNode_; //OPTIMIZE - used only once
     std::vector<int> nodeOffsetI = subProb_[*cliqI].getNodeOffset(); //OPTIMIZE - used only once
#else
     std::vector<short> nLabelI = subProb_[cliqJ].getNodeLabel();
     int sizCliqI = subProb_[cliqJ].sizCliq_; //OPTIMIZE - used only once
     int cliqOffsetI = subProb_[cliqJ].getCliqOffset(); //OPTIMIZE - used only once
     std::vector<int> memNodeI = subProb_[cliqJ].memNode_; //OPTIMIZE - used only once
     std::vector<int> nodeOffsetI = subProb_[cliqJ].getNodeOffset(); //OPTIMIZE - used only once
#endif

     double f1 = 0, f2 = 0;

     int blkRow = 0;

     if (*cliqI == cliqJ) {
      //std::cout<<"CLIQUE "<<cliqJ<<std::endl;
      for (int elemI = 0; elemI != sizCliqJ; ++elemI) {
       for (int labelI = 0; labelI != nLabelI[elemI]; ++labelI) {

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

        double f2NumTermI, f2NumTermJ;

        if (curJNode == memNodeI[elemI]) {

         f2NumTermI = f2LabelSum[unaryOffset_[curJNode] + labelI];

         if (labelI == labelJ) {
          f2 = f2NumTermI*(1 - f2NumTermI);
         }
         else {
          f2NumTermJ = f2LabelSum[unaryOffset_[curJNode] + labelJ];

          f2 = -1*f2NumTermI*f2NumTermJ;
         }
        }
        else {
         f2 = 0;
        }

        double curVal =  tau_*(f1 + f2);

        int curI = cliqOffsetI + nodeOffsetI[elemI] + labelI;

        if (curI == curJ) {
         curVal += dampLambda_;
         hessDiag_[curJ] = curVal; //for inc. Cholesky
        }
        else { //track off diagonal elements
         ++totOffDiag;
         if (curVal < 0) {
          ++totNegOffDiagH1;
         }
        }

        //populate compressed column data structure for inc. Cholesky preconditioner
        if (curI > curJ) {
         ++nNonZeroLower_; //inc. Cholesky
         nzHessLower_.push_back(curVal);
         hessRowInd_.push_back(curI+1);
        }

        eigenHess_.insert(curI,curJ) = curVal;
        ++elemCnt; //####

        //std::cout<<tau_*f2<<" ";

        //make preconditioner PD by adding sufficient damping.
        if ((curI == curJ) && (dampLambda_ < 0.001)) {
         curVal += 0.001;
        }

        cliqBlk(blkRow,blkCol) = curVal;

//        std::cout<<curI<<" "<<curJ<<" "<<f1<<" "<<f2<<std::endl;
        ++blkRow;        
       } //labelI
      } //elemI
      //std::cout<<std::endl;
     } //if *cligI == cliqJ
     else {
      for (int elemI = 0; elemI != sizCliqI; ++elemI) {
       if (curJNode == memNodeI[elemI]) {
        for (int labelI = 0; labelI != nLabelI[elemI]; ++labelI) {

         double f2NumTermI, f2NumTermJ;

         f2NumTermI = f2LabelSum[unaryOffset_[curJNode] + labelI];

         if (labelI == labelJ) {
          f2 = f2NumTermI*(1 - f2NumTermI);
         }
         else {
          f2NumTermJ = f2LabelSum[unaryOffset_[curJNode] + labelJ];

          f2 = -1*f2NumTermI*f2NumTermJ;
         }

         double curVal =  tau_*f2;

         int curI = cliqOffsetI + nodeOffsetI[elemI] + labelI;

//         h2Pattern[curI][curJ] = curJNode+1; //h2PatFill[cliqJ][elemJ]; //####
//         h2Pattern[curJ][curI] = curJNode+1;

         ++totOffDiag;
         if (curVal < 0) {
          ++totNegOffDiagH2;
         }

         eigenHess_.insert(curI,curJ) = curVal;
         ++elemCnt; //####

         ++nNonZeroLower_; //inc. Cholesky
         nzHessLower_.push_back(curVal);
         hessRowInd_.push_back(curI+1);
//         std::cout<<curI<<" "<<curJ<<" "<<f1<<" "<<f2<<std::endl;
        }
       }
      }
     }
    }
    } //for iterating through cliqNeigh
    ++blkCol;
   } // for labelJ
  } //for elemJ
#endif
  Eigen::MatrixXd cliqBlkInv = cliqBlk.inverse();
  blkJacob_[cliqJ] = cliqBlkInv;
 } //for cliqJ
#endif

 //hessColPtr_[nDualVar_] = nNonZeroLower_; //inc. Cholesky
 hessColPtr_[nDualVar_] = nNonZeroLower_+1; //inc. Cholesky

// std::cout<<"total off-diagonal elements "<<totOffDiag<<" total negative H1 "<<totNegOffDiagH1<<" total negative H2 "<<totNegOffDiagH2<<std::endl;

// std::cout<<"no. of non-zeros in hessian "<<elemCnt<<std::endl; //####

 if (isnan(curEnergy_)) {
  std::cout<<"ENERGY INDEED NAN!"<<std::endl;
  return -1;
 }

 return 0;
}

int dualSys::popSparseGradEnergy() {
 int totOffDiag = 0, totNegOffDiagH1 = 0, totNegOffDiagH2 = 0;

 double f2DenExpMax;
 std::vector<double> f2LabelSum(numLabTot_);

 curEnergy_ = 0;

 unsigned long long elemCnt = 0; //check non-zeros of hessian

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

  curEnergy_ += (1/tau_)*log(f2Den) + f2DenExpMax;
 } //for curNode

 double f1Den;

#if 1
// #pragma omp parallel for
 for (int cliqJ = 0; cliqJ < nCliq_; ++cliqJ) {
  int nCliqLab = subProb_[cliqJ].nCliqLab_;
  int sizCliqJ = subProb_[cliqJ].sizCliq_;
  std::vector<short> nLabelJ = subProb_[cliqJ].getNodeLabel();
  std::vector<std::pair<std::vector<int>,double> > sparseCEnergy = subProb_[cliqJ].getSparseCEnergy();
  std::vector<double> dualVar = subProb_[cliqJ].getDualVar();
  std::vector<int> nodeOffsetJ = subProb_[cliqJ].getNodeOffset();
  int cliqOffsetJ = subProb_[cliqJ].getCliqOffset();
  std::set<int> cliqNeigh = subProb_[cliqJ].cliqNeigh_;
  std::vector<int> memNodeJ = subProb_[cliqJ].memNode_;

  int subDualSiz = dualVar.size();

  std::vector<double> nodeLabSum(subDualSiz);

  Eigen::MatrixXd cliqBlk(subDualSiz,subDualSiz);

  double expVal;

  f1Den = 0;

  std::vector<double> f1DenExp(nCliqLab);

  double f1DenExpMax = 0;

  for (std::vector<std::pair<std::vector<short>,double> >::iterator iSparse = sparseCEnergy.begin(); iSparse != sparseCEnergy.end(); ++iSparse) {
   std::vector<short> cliqLab = (*iSparse).first;

   double nodeSum = 0;

   for (int j = 0; j != sizCliqJ; ++j) {
    nodeSum += dualVar[nodeOffsetJ[j] + cliqLab[j]];
   }

   f1DenExp[i] = (*iSparse).second - nodeSum; //(*iSparse).second is clique energy

   if (i == 0) {
    f1DenExpMax = f1DenExp[i];
   }
   else {
    if (f1DenExp[i] > f1DenExpMax) {
     f1DenExpMax = f1DenExp[i];
    }
   }
  }

  int iCliqLab = 0;

  for (std::vector<std::pair<std::vector<short>,double> >::iterator iSparse = sparseCEnergy.begin(); iSparse != sparseCEnergy.end(); ++iSparse) {
   expVal = exp(tau_*(f1DenExp[iCliqLab] - f1DenExpMax));
   myUtils::checkRangeError(expVal);
   f1Den += expVal;

   std::vector<short> cliqLab = (*iSparse).first;

   for (int j = 0; j != sizCliqJ; ++j) {
    nodeLabSum[nodeOffsetJ[j] + cliqLab[j]] += expVal;
   }

   ++iCliqLab;
  }

  for (std::vector<double>::iterator nodeLabIter = nodeLabSum.begin(); nodeLabIter != nodeLabSum.end(); ++nodeLabIter) {
   *nodeLabIter /= f1Den;
  }

  curEnergy_ += (1/tau_)*log(f1Den) + f1DenExpMax;

  int blkCol = 0;

#if 1
  for (int elemJ = 0; elemJ != sizCliqJ; ++elemJ) {
   for (int labelJ = 0; labelJ != nLabelJ[elemJ]; ++labelJ) {
    int curJ = cliqOffsetJ + nodeOffsetJ[elemJ] + labelJ;

    int curJNode = memNodeJ[elemJ];

    gradient_[curJ] = -1*nodeLabSum[nodeOffsetJ[elemJ] + labelJ] + f2LabelSum[unaryOffset_[curJNode] + labelJ];
   } // for labelJ
  } //for elemJ
#endif
 } //for cliqJ
#endif

 if (isnan(curEnergy_)) {
  std::cout<<"ENERGY INDEED NAN!"<<std::endl;
  return -1;
 }

 return 0;
}

int dualSys::performSparseLineSearch() {
 std::vector<double> z(nDualVar_);

 std::vector<double> oriNewtonStep = newtonStep_;

 myUtils::addArray<double>(dualVar_,newtonStep_,z,nDualVar_);

 double iterEnergy = computeSparseEnergy(z);

 int lsCnt = 0;

// std::cout<<"about to enter ls: LHS energy "<<iterEnergy<<" RHS energy "<<curEnergy_<<" "<<lsC_<<" "<<myUtils::dotVec(gradient_,newtonStep_,nDualVar_)<<" "<<lsTol_<<std::endl;

 while (iterEnergy > curEnergy_ + lsC_*(myUtils::dotVec<double>(gradient_,newtonStep_,nDualVar_)) + lsTol_) {
  ++lsCnt;

  lsAlpha_ *= lsRho_;

  myUtils::scaleVector<double>(lsAlpha_,oriNewtonStep,newtonStep_,nDualVar_);
  myUtils::addArray<double>(dualVar_,newtonStep_,z,nDualVar_);

  iterEnergy = computeSparseEnergy(z);

  double rhsEnergy = curEnergy_ + lsC_*(myUtils::dotVec<double>(gradient_,newtonStep_,nDualVar_)) + lsTol_;

//  std::cout<<"inside line search: lhs "<<iterEnergy<<" rhs "<<rhsEnergy<<std::endl;
 }

 lsAlpha_ = 1;

 return lsCnt;
}

int dualSys::solveNewton()
{
 bool sparseFlag = true;

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

 double gradBasedDamp;
 bool gradDampFlag = true;

 dampLambda_ = 10; //####

 Eigen::VectorXd eigenGuess = Eigen::VectorXd::Zero(nDualVar_);

 while ((contIter) && (cntIter < maxIter_)) {
  double tFull = myUtils::getTime();

  ++cntIter;
  ++cntIterTau;

  eigenHess_.reserve(reserveHess_);

  double tGHE = myUtils::getTime(); //time to compute gradient, hessian and current energy

  if (sparseFlag) {
   popSparseGradHessEnergy();
  }
  else {
   popGradHessEnergy();
  }

  double gradNorm = myUtils::norm<double>(gradient_, nDualVar_);

  if (cntIter == 1) { 
   gradBasedDamp = gradNorm/6;
  }
  else {
   gradBasedDamp = gradBasedDamp;
  }

  if ((annealIval_ != -1) && (gradNorm < gradBasedDamp) && (tau_ < tauMax_)) {
//   std::cout<<"grad norm "<<gradNorm<<" grad based damp "<<gradBasedDamp<<std::endl;
   cntIterTau = 1;
   tau_ *= 2;
   gradDampFlag = true; //earlier, it was 3 ####
   dampLambda_ = 10; //####

   eigenHess_.setZero();
   eigenHess_.reserve(reserveHess_);

   if (sparseFlag) {
    popSparseGradHessEnergy();
   }
   else {
    popGradHessEnergy();
   }

//  }
//  if (gradDampFlag) {
   gradNorm = myUtils::norm<double>(gradient_, nDualVar_); 
   gradBasedDamp = gradNorm/6;

   if (gradBasedDamp < dampTol_) {
    gradBasedDamp = dampTol_;
   }
//   gradDampFlag = false; 
  }

  int gradMaxInd = myUtils::argmaxAbs<double>(gradient_,0,nDualVar_);

  double gradMax = std::abs(gradient_[gradMaxInd]);

  std::cout<<"ITERATION "<<cntIter<<" starts, with tau "<<tau_<<" and Gradient threshold "<<gradBasedDamp<<"."<<std::endl;
//  std::cout<<"solveNewton: populated gradient, hessian and energy "<<(myUtils::getTime() - tGHE)<<" seconds."<<std::endl;
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

   //custom CG implementation
   for (std::size_t i = 0; i < nDualVar_; ++i) {
    oriGrad[i] = gradient_[i];
    eigenGrad_[i] = gradient_[i];
   }

   double tCG = myUtils::getTime();
   std::vector<Eigen::VectorXd> stepBackVec; //for back tracking
   eigenStep = eigenGuess;
   solveCG(eigenStep, stepBackVec, cntIterTau % 10);

   std::cout<<"CG took "<<myUtils::getTime() - tCG<<" seconds."<<std::endl;

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

   Eigen::VectorXd eigenInter = eigenHess_.selfadjointView<Eigen::Lower>()*eigenStep;

   double interValOne = eigenStep.dot(eigenInter);

   interValOne *= 0.5;

   double interValTwo = eigenStep.dot(oriGrad);

   double nxtApproxEnergyDiff = interValOne + interValTwo; //diff. wrt current energy

   std::vector<double> z(nDualVar_);

   myUtils::addArray<double>(dualVar_,newtonStep_,z,nDualVar_);

   double nxtEnergy;

   if (sparseFlag) {
    nxtEnergy = computeSparseEnergy(z);
   }
   else {
    nxtEnergy = computeEnergy(z);
   }

   double rho = (nxtEnergy - curEnergy_)/nxtApproxEnergyDiff;

   if ((isnan(rho)) || (isinf(rho))) {
    std::cout<<"rho debug: nxtApproxEnergyDiff "<<nxtApproxEnergyDiff<<" nxtEnergy "<<nxtEnergy<<" curEnergy_ "<<curEnergy_<<std::endl;
   }

   double eta_1 = 0.25, eta_2 = 0.5, eta_3 = 0.9, sigma_1 = 2, sigma_2 = 0.5, sigma_3 = 0.25, eta_0 = 0.0001;

   //updating damping lambda inspired by "Newton's method for large-scale optimization" by Bouaricha et al

   //std::cout<<"next energy "<<nxtEnergy<<" "<<" current energy "<<curEnergy_<<" next approx energy diff "<<nxtApproxEnergyDiff<<" interValOne "<<interValOne<<" interValTwo "<<interValTwo<<std::endl;
 
   int newtonMaxInd = myUtils::argmaxAbs<double>(newtonStep_,0,nDualVar_);
   double stepNormOld = 0, stepNormNew = 0;
//  std::cout<<"solveNewton: before line search newton step l-infinity norm: "<<newtonStep_[newtonMaxInd]<<". Euclidean norm: "<<newtonStepNorm<<std::endl;

   double tLS = myUtils::getTime(); //time to perform line search

   int lsCnt = -1;

   if (rho <= eta_0) {
    stepNormOld = myUtils::norm<double>(newtonStep_, nDualVar_);

    std::vector<double> oriNewton = newtonStep_;
//    lsCnt = performCGBacktrack(stepBackVec);

//    std::vector<double> resetStep = newtonStep_;

//    newtonStep_ = oriNewton;
    lsCnt = performLineSearch(); //perform line search only when rho <= eta_0

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

   newtonMaxInd = myUtils::argmaxAbs<double>(newtonStep_,0,nDualVar_);
//  std::cout<<"solveNewton: after line search newton step l-infinity norm: "<<newtonStep_[newtonMaxInd]<<". Euclidean norm: "<<newtonStepNorm<<std::endl;
   std::cout<<"solveNewton: performed line search "<<lsCnt<<" times, "<<(myUtils::getTime() - tLS)<<" seconds."<<std::endl;

   myUtils::addArray<double>(dualVar_,newtonStep_,dualVar_,nDualVar_);

//   int dualVarInd = myUtils::argmaxAbs<double>(dualVar_,0,nDualVar_);
//   double dualNorm = myUtils::norm<double>(dualVar_, nDualVar_);
//   std::cout<<"solveNewton: dual l-infinity norm: "<<dualVar_[dualVarInd]<<". Euclidean norm: "<<dualNorm<<std::endl;

   distributeDualVars();

   eigenHess_.setZero(); //change logic to insert in first iteration and coeffRef in subsequent ones.
  }
  else if (tau_ == tauMax_) {
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
  recoverFeasPrimal();
  recoverMaxPrimal(primalConsist_);

  std::cout<<std::fixed;
  std::cout<<std::setprecision(6);

  std::cout<<"Fractional primal energy: "<<compSmoothPrimalEnergy()<<" Smooth dual energy: "<<curEnergy_<<std::endl;

  recoverMaxPrimal(primalFrac_);
  std::cout<<" Integral primal energy (directly from dual): "<<compIntPrimalEnergy()<<" Non-smooth dual energy: "<<computeNonsmoothDualEnergy()<<std::endl;

//  std::cout<<"Computing energies took "<<myUtils::getTime() - tEnergy<<" seconds."<<std::endl;
//  std::cout<<"NOTE: ITERATION "<<cntIter<<" took "<<(tEnergy - tFull)<<" seconds. The following figure includes unnecessary energy computation."<<std::endl;

  tCompEnergies += myUtils::getTime() - tEnergy;
 }
#endif

  std::cout<<"ITERATION "<<cntIter<<" took "<<(myUtils::getTime() - tFull)<<" seconds."; 
 }

 recoverFracPrimal();
 recoverFeasPrimal();
 recoverMaxPrimal(primalConsist_);

 tau_ = 100000; //####

 std::cout<<std::fixed;
 std::cout<<std::setprecision(6);

 std::cout<<"solveNewton: Fractional primal energy: "<<compSmoothPrimalEnergy()<<std::endl;
 std::cout<<"solveNewton: Smooth dual energy: "<<computeEnergy(dualVar_)<<std::endl;
 std::cout<<"solveNewton: Integral primal energy (feasible): "<<compIntPrimalEnergy()<<std::endl;

 recoverMaxPrimal(primalFrac_);
 std::cout<<"solveNewton: Integral primal energy (directly from dual): "<<compIntPrimalEnergy()<<std::endl;
 std::cout<<"solveNewton: Non-smooth dual energy: "<<computeNonsmoothDualEnergy()<<std::endl;

 std::cout<<"Total time spent on computing energies in each iteration: "<<tCompEnergies<<std::endl;

 return 0;
}

double dualSys::computeSparseEnergy(std::vector<double> var)
{
 double expVal = 0;
 double f1 = 0;

 for (int i = 0; i != nCliq_; ++i) {
  short sizCliq = subProb_[i].sizCliq_;
  std::vector<short> nLabel = subProb_[i].getNodeLabel();
  std::vector<std::pair<int,double> > sparseCEnergy = subProb_[i].getSparseCEnergy();
  std::vector<int> nodeOffset = subProb_[i].getNodeOffset();
  double f1Cliq = 0;
  double f1Max = 0;
  double* f1Whole = new double[subProb_[i].nCliqLab_];

  int iCliqLab = 0;

  for (std::vector<std::pair<std::vector<short>,double> >::iterator iSparse = sparseCEnergy.begin(); iSparse != sparseCEnergy.end(); ++iSparse) {
   std::vector<short> cliqLab = (*iSparse).first;

   double dualSum = 0;

   for (int k = 0; k != sizCliq; ++k) {
    dualSum += var[subProb_[i].getCliqOffset() + nodeOffset[k] + cliqLab[k]];
    if (isnan(dualSum)) {
     std::cout<<"F1 DUAL SUM IS NAN! dual variable is "<<var[subProb_[i].getCliqOffset() + nodeOffset[k] + cliqLab[k]]<<std::endl;
    }
  }

   f1Whole[iCliqLab] = ((*iSparse).second - dualSum); //(*iSparse).second is the clique energy
 
   ++iCliqLab;
  }

  f1Max = *std::max_element(f1Whole,f1Whole + subProb_[i].nCliqLab_);

  for (int j = 0; j != subProb_[i].nCliqLab_; ++j) {
   expVal =  exp(tau_*(f1Whole[j] - f1Max));

   myUtils::checkRangeError(expVal);

   f1Cliq += expVal;
  }

  f1 += (log(f1Cliq) + tau_*f1Max);

  delete[] f1Whole;
 }

 double f2 = 0;

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
   }

   f2Whole[j] = uEnergy_[unaryOffset_[i] + j] + dualSum;
  }

  f2Max = *std::max_element(f2Whole, f2Whole + nLabel);

  for (int j = 0; j != nLabel; ++j) {

   expVal = exp(tau_*(f2Whole[j] - f2Max));

   myUtils::checkRangeError(expVal);

   f2Node += expVal;
  }
 
  f2 += (log(f2Node) + tau_*f2Max);
 
  delete[] f2Whole;
 }

 double returnVal = (1/tau_)*(f1 + f2);
 
 if (isinf(returnVal)) {
  std::cout<<"computeEnergy: energy is INF"<<std::endl;
 }
 else if (isnan(returnVal)) {
  std::cout<<"computeEnergy: energy is NAN!"<<std::endl;
 }

 return returnVal;
}

double dualSys::computeSparseNonsmoothDualEnergy()
{
 double f1 = 0;
  
 for (int i = 0; i != nCliq_; ++i) {
  //std::vector<double> cEnergy = subProb_[i].getCE();
  //std::vector<double> cEnergy = cEnergy2DVec_[cliqOffset_[i]];
  std::vector<std::pair<int,double> > sparseCEnergy = subProb_[i].getSparseCEnergy();
  std::vector<int> nodeOffset = subProb_[i].getNodeOffset();
  std::vector<short> nLabel = subProb_[i].getNodeLabel();
  int sizCliq = subProb_[i].sizCliq_;

  double f1Cliq = -1.0*pow(10,10);

  for (std::vector<std::pair<std::vector<short>,double> >::iterator iSparse = sparseCEnergy.begin(); iSparse != sparseCEnergy.end(); ++iSparse) {
   double dualSum = 0;

   std::vector<short> cliqLab = (*iSparse).first;

   for (int k = 0; k != sizCliq; ++k) {
    dualSum += subProb_[i].getDualVar()[nodeOffset[k] + cliqLab[k]];
   }

   if (((*iSparse).second - dualSum) > f1Cliq) {
    f1Cliq = (*iSparse).second - dualSum;
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

void dualSys::recoverSparseFracPrimal()
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

 for (int i = 0; i != nCliq_; ++i) {
  double argMax = 0;
  int nCliqLab = subProb_[i].nCliqLab_;
  std::vector<double> argVal(nCliqLab);
  std::vector<short> nLabel = subProb_[i].getNodeLabel();
  std::vector<int> nodeOffset = subProb_[i].getNodeOffset();
  int sizCliq = subProb_[i].sizCliq_;
  std::vector<std::pair<std::vector<short>,double> > sparseCEnergy = subProb_[i].getSparseCEnergy();

  int iCliqLab = 0;

  for (std::vector<std::pair<std::vector<short>,double> >::iterator iSparse = sparseCEnergy.begin(); iSparse != sparseCEnergy.end(); ++iSparse) {
   double dualSum = 0;

   std::vector<short> cliqLab = (*iSparse).first;

   for (int k = 0; k != sizCliq; ++k) {
    dualSum += subProb_[i].getDualVar()[nodeOffset[k] + cliqLab[k]];
   }

   argVal[iCliqLab] = (*iSparse).second - dualSum;

   if (iCliqLab == 0) {
    argMax = argVal[iCliqLab];
   }
   else {
    if (argVal[iCliqLab] > argMax) {
     argMax = argVal[iCliqLab];
    }
   }

   ++iCliqLab;
  }

  double cliqNorm = 0;

  for (int j = 0; j != nCliqLab; ++j) {
   expVal = exp(tau_*(argVal[j] - argMax));
   myUtils::checkRangeError(expVal);
   subProb_[i].primalCliqFrac_[j] = expVal;
   cliqNorm += expVal;
  }

  for (int j = 0; j != nCliqLab; ++j) {
   fracVal =  (1/cliqNorm)*subProb_[i].primalCliqFrac_[j];
   subProb_[i].primalCliqFrac_[j] = fracVal;
  }
 }
}

void dualSys::recoverSparseFeasPrimal()
{
 std::vector<std::vector<double> > cliqMargSum(nCliq_);

 for (int i = 0; i != nCliq_; ++i) {
  int sizCliq = subProb_[i].sizCliq_;
  int subDualSiz = subProb_[i].getDualSiz();
  std::vector<int> nodeOffset = subProb_[i].getNodeOffset();
  std::vector<short> nLabel = subProb_[i].getNodeLabel();
  int nCliqLab = subProb_[i].nCliqLab_;
  std::vector<std::pair<int,double> > sparseCEnergy = subProb_[i].getSparseCEnergy();

  std::vector<double> curMargSum(subDualSiz, 0);

  int iCliqLab = 0;

  for (std::vector<std::pair<std::vector<short>,double> >::iterator iSparse = sparseCEnergy.begin(); iSparse != sparseCEnergy.end(); ++iSparse) {
   std::vector<short> cliqLab = (*iSparse).first;

   for (int k = 0; k != sizCliq; ++k) {
    for (int l = 0; l != nLabel[k]; ++l) {
     if (cliqLab[k] == l) {
      curMargSum[nodeOffset[k] + l] += subProb_[i].primalCliqFrac_[iCliqLab];
     }
    }
   }

   ++iCliqLab;
  }
  cliqMargSum[i] = curMargSum;
 }

 primalConsist_.resize(uEnergy_.size());

 for (int i = 0; i != nNode_; ++i) {
  int nLabel = nLabel_[i];
  for (int j = 0; j != nLabel; ++j) {
   double cliqSum = 0;
   double cliqDiv = 0;
   for (std::vector<int>::iterator k = cliqPerNode_[i].begin(); k != cliqPerNode_[i].end(); ++k) {
    int nodeInd = myUtils::findNodeIndex(subProb_[*k].memNode_, i);

    std::vector<short> cliqNodeLabel = subProb_[*k].getNodeLabel();

    double margNorm = 1;
    for (int l = 0; l != subProb_[*k].sizCliq_; ++l) {
     if (l != nodeInd) {
      margNorm *= cliqNodeLabel[l];
     }
    }
    margNorm = 1/margNorm;

    cliqSum += margNorm*cliqMargSum[*k][subProb_[*k].getNodeOffset()[nodeInd] + j];
    cliqDiv += margNorm;
   }
   primalConsist_[unaryOffset_[i] + j] = (1.0/(1.0 + cliqDiv))*(primalFrac_[unaryOffset_[i] + j] + cliqSum);
  }
 }

 double lambd = 0.0;

 for (int i = 0; i != nCliq_; ++i) {
  int sizCliq = subProb_[i].sizCliq_;
  int subDualSiz = subProb_[i].getDualSiz();
  std::vector<int> nodeOffset = subProb_[i].getNodeOffset();
  std::vector<short> nLabel = subProb_[i].getNodeLabel();
  int nCliqLab = subProb_[i].nCliqLab_;

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

   if (subProb_[i].primalCliqConsist_[j] < 0) {
    if (lambd < subProb_[i].primalCliqConsist_[j]/denTerm) {
     lambd = subProb_[i].primalCliqConsist_[j]/denTerm;
    }
   }
   else if (subProb_[i].primalCliqConsist_[j] > 1) {
    if (lambd < (subProb_[i].primalCliqConsist_[j] - 1.0)/denTerm) {
     lambd = (subProb_[i].primalCliqConsist_[j] - 1.0)/denTerm;
    }
   }
  }
 }

 for (int i = 0; i != nNode_; ++i) {
  int nLabel = nLabel_[i];
  double nodeLambdTerm = lambd/nLabel;

  for (int j = 0; j != nLabel; ++j) {
   primalConsist_[unaryOffset_[i] + j] = (1.0 - lambd)*primalConsist_[unaryOffset_[i] + j] + nodeLambdTerm;
  }
 }

 for (int i = 0; i != nCliq_; ++i) {
  double cliqLambdTerm = lambd/subProb_[i].nCliqLab_;

  for (int j = 0; j != subProb_[i].nCliqLab_; ++j) {
   subProb_[i].primalCliqConsist_[j] = (1.0 - lambd)*subProb_[i].primalCliqConsist_[j] + cliqLambdTerm;
  }
 }
}

double dualSys::compIntPrimalEnergy()
{
 double energy = 0;
 for (int i = 0; i != nCliq_; ++i) {
  //energy += subProb_[i].getCE()[subProb_[i].primalCliqMax_];
  energy += cEnergy2DVec_[cliqOffset_[i]][subProb_[i].primalCliqMax_];
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
   logVal = log(subProb_[i].primalCliqFrac_[j]);

   if ((errno == ERANGE) || (isnan(logVal))) {
    entropyTerm = 0;
    errno = 0;
   }
   else {
    entropyTerm = subProb_[i].primalCliqFrac_[j]*logVal;
   }

   //energy += subProb_[i].getCE()[j]*subProb_[i].primalCliqFrac_[j] - (1/tau_)*entropyTerm;
   energy += cEnergy2DVec_[cliqOffset_[i]][j]*subProb_[i].primalCliqFrac_[j] - (1/tau_)*entropyTerm;
   if ((isnan(energy)) && (probCliqCnt == 0)) {
    ++probCliqCnt;
    std::cout<<"node "<<i<<" cEnergy "<<cEnergy2DVec_[cliqOffset_[i]][j]<<" primalCliqFrac_ "<<subProb_[i].primalCliqFrac_[j]<<" entropy term "<<entropyTerm<<std::endl;
   }
  }
 }

 for (int i = 0; i != nNode_; ++i) {
  int nLabel = nLabel_[i];
  for (int j = 0; j != nLabel; ++j) {
   logVal = log(primalFrac_[unaryOffset_[i] + j]);

   if ((errno == ERANGE) || (isnan(logVal))) {
    entropyTerm = 0;
    errno = 0;
   }
   else {
    entropyTerm = primalFrac_[unaryOffset_[i] + j]*logVal;
   }

   energy += uEnergy_[unaryOffset_[i] + j]*primalFrac_[unaryOffset_[i] + j] - (1/tau_)*entropyTerm;
   if ((isnan(energy)) && (probNodeCnt == 0)) {
    ++probNodeCnt;
    std::cout<<"node "<<i<<" uEnergy "<<uEnergy_[unaryOffset_[i] + j]<<" primalFrac "<<primalFrac_[unaryOffset_[i] + j]<<" entropy term "<<entropyTerm<<std::endl;
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

 bool build;
 int info, mssg;

 if (precondFlag_ == 2) { //create Incomplete Cholesky data-structures
  inCholLow_ = new double[nzInChol];
  inCholRowInd_ = new int[nzInChol];

  double tInChol = myUtils::getTime();
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
 residual = eigenHess_.selfadjointView<Eigen::Lower>()*iterStep + eigenGrad_; 

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
   for (int i = 0; i != nDualVar_; ++i) {
    resVec_[i] = residual[i];
   }
   precnd_(&numVarQN_, &numPairQN_, &iopQN_, &newtonIter, &cntIter, sVec_, yVec_, resVec_, prVec_, workVecQN_, &lwQN_, iWorkVecQN_, &liwQN_, &build, &info, &mssg);
   for (int i = 0; i != nDualVar_; ++i) {
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
  forceTerm = 0.0001;
 }
 else if (tau_ < tauMax_/2) {
  forceTerm = 0.0001;
 }
 else {
  forceTerm = 0.0001;
 }
#endif

 double residualCond = (forceTerm < sqrt(eigenGrad_.norm())) ? forceTerm*eigenGrad_.norm():sqrt(eigenGrad_.norm())*eigenGrad_.norm();

 if (residualCond < 1e-4) {
  residualCond = 1e-4;
 }

 bool exitCond = false;

 double totPrecondTime = 0;

 int backTrackPow = 1;
 double backTrackIndex = 1.3;

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
//   double measDirecProd = myUtils::getTime();
//   Eigen::VectorXd hessDirProd = eigenHess_*direc;
   Eigen::VectorXd hessDirProd = eigenHess_.selfadjointView<Eigen::Lower>()*direc;

//   std::cout<<"direct hessian vector product "<<myUtils::getTime() - measDirecProd<<std::endl;  

#if 0 //####
   std::vector<double> direcVec(nDualVar_);
//   std::cout<<"multiplying vector"<<std::endl;
   for (std::size_t i = 0; i != nDualVar_; ++i) {
    direcVec[i] = direc[i];
//    std::cout<<direc[i]<<std::endl;
   }

   std::ofstream hessDirProdFile("hessDirProdFile.txt");
   std::ofstream direcFile("direcFile.txt");

   for (std::size_t i = 0; i != nDualVar_; ++i) {
    hessDirProdFile<<hessDirProd[i]<<std::endl;
    direcFile<<direc[i]<<std::endl;
   }

   hessDirProdFile.close();
   direcFile.close();

   std::vector<double> checkDirec(nDualVar_), checkHessDirProd(nDualVar_);

   for (std::size_t i = 0; i != nDualVar_; ++i) {
    checkDirec[i] = direc[i];
    checkHessDirProd[i] = hessDirProd[i];
   }

   double maxDirec = *std::max_element(checkDirec.begin(), checkDirec.end());
   double minDirec = *std::min_element(checkDirec.begin(), checkDirec.end());
   double maxHessDirProd = *std::max_element(checkHessDirProd.begin(), checkHessDirProd.end());
   double minHessDirProd = *std::min_element(checkHessDirProd.begin(), checkHessDirProd.end());

   std::cout<<"max. direction "<<maxDirec<<" min. direction "<<minDirec<<" max. HessDirProd "<<maxHessDirProd<<" min. HessDirProd "<<minHessDirProd<<std::endl;
#endif

#if 0 //#### Complex Step Differentiation
   std::vector<double> approxProd(nDualVar_);

   double measCSD = myUtils::getTime();
   matVecMult(*this,direcVec,approxProd);

   std::cout<<"csd based "<<myUtils::getTime() - measCSD<<std::endl;

   int agreeCnt = 0;
   int nanCnt = 0;
   double checkTol = 1e-4;

//   std::ofstream hessVecDebug("f1HessVecDebug.txt");

//   std::cout<<"approx. product"<<std::endl;
   for (std::size_t i = 0; i != nDualVar_; ++i) {
  //  std::cout<<approxProd[i]<<std::endl;
    if ((!isnan(approxProd[i])) && (!isinf(approxProd[i]))) {
     if ((hessDirProd[i] > approxProd[i] - checkTol) && (hessDirProd[i] < approxProd[i] + checkTol)) {
      ++agreeCnt;
     }
    }
    else {
     ++nanCnt;
    }
//    std::cout<<"mult. vector "<<direcVec[i]<<" after mult "<<hessDirProd[i]<<" by approx "<<approxProd[i]<<std::endl;
//    hessVecDebug<<approxProd[i]<<std::endl;
   }

//   hessVecDebug.close();

   std::cout<<"no. of dual variables "<<nDualVar_<<" agree count "<<agreeCnt<<" nan count "<<nanCnt<<std::endl;
#endif

   alpha = resDotPreRes/(direc.transpose()*hessDirProd);
   iterStep += alpha*direc;

#if 0
   std::vector<double> iterVec;
   for (int debugI = 0; debugI != nDualVar_; ++debugI) {
    iterVec.push_back(iterStep[debugI]);
   }
   std::vector<double> z(nDualVar_);
   myUtils::addArray<double>(dualVar_,iterVec,z,nDualVar_);
   //std::cout<<"solveCG: current energy is "<<computeEnergy(z)<<std::endl;
#endif

#if 0
   if (cntIter == ceil(pow(backTrackIndex,backTrackPow))) {
    iterStepVec.push_back(iterStep);
    ++backTrackPow;
    if (ceil(pow(backTrackIndex,backTrackPow)) == cntIter) {
     ++backTrackPow;
    }
   }
#endif

   Eigen::VectorXd nxtRes;
   if (cntIter % 50 == 0) {
    nxtRes = eigenHess_.selfadjointView<Eigen::Lower>()*iterStep - b;
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
     for (int i = 0; i != nDualVar_; ++i) {
      resVec_[i] = nxtRes[i];
      sVec_[i] = direc[i];
      yVec_[i] = hessDirProd[i]; 
     }
     precnd_(&numVarQN_, &numPairQN_, &iopQN_, &newtonIter, &cntIter, sVec_, yVec_, resVec_, prVec_, workVecQN_, &lwQN_, iWorkVecQN_, &liwQN_, &build, &info, &mssg);
     for (int i = 0; i != nDualVar_; ++i) {
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

 for (int i = 0; i != nDualVar_; ++i) {
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
 int nzInChol = nNonZeroLower_ + orderMat*p;

 int* iwa = new int[3*orderMat];
 double* wa1 = new double[orderMat];
 double* wa2 = new double[orderMat];

 double* nzHessLower = new double[nNonZeroLower_];
 int* hessRowInd = new int[nNonZeroLower_];

 for (std::size_t i = 0; i != nNonZeroLower_; ++i) {
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

 for (std::size_t i = 0; i != orderMat; ++i) {
  r[i] = residual[i];
 }

 Eigen::VectorXd precondRes(orderMat);

 dstrsol_(&orderMat,inCholLow_,inCholDiag_,inCholColPtr_,inCholRowInd_,r,&inTask);
 dstrsol_(&orderMat,inCholLow_,inCholDiag_,inCholColPtr_,inCholRowInd_,r,&outTask);

 for (std::size_t i = 0; i != orderMat; ++i) {
  precondRes[i] = r[i];
 }

 delete[] r;

 return precondRes;
}

int dualSys::solveFista() {
 double gradBasedDamp;
 bool gradDampFlag = true;

 double L0 = 10;

 double L = L0;
 double beta = 1.1;
 double t_k = 1, t_prev;

 std::vector<double> x_k(nDualVar_,0);
 std::vector<double> x_prev = x_k;
 std::vector<double> diffVec(nDualVar_);

 dualVar_ = x_k;

 distributeDualVars();

 t_prev = t_k;

 bool contIter = true;

 int cntIter = 0, cntIterTau = 0;

 double totEnergyTime = 0;

 while (contIter) {
  double tFull = myUtils::getTime();  

  ++cntIter;
  ++cntIterTau;
  
  popGradEnergy();

  double gradNorm = myUtils::norm<double>(gradient_, nDualVar_);

  if (gradDampFlag) {
   gradBasedDamp = gradNorm/6; //earlier, it was 3 ####
   gradDampFlag = false;
  }

  if ((annealIval_ != -1) && (gradNorm < gradBasedDamp) && (tau_ < tauMax_)) {
   tau_ *= 2;
   gradDampFlag = true;
   L = L0; //resetting L ####
  }

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

  if ((gradMax < 0.001) || (cntIter > maxIter_)) {
   contIter = false;
  }

  double invL = -1/L;

  myUtils::scaleVector<double>(invL,gradient_,x_k,nDualVar_);

  //std::cout<<"norm should be 0: "<<myUtils::norm(x_k,nDualVar_)<<std::endl;

  //std::cout<<"energy at y: "<<computeEnergy(dualVar_)<<" energy at x_k "<<computeEnergy(x_k)<<" energy computed by routine "<<curEnergy_<<std::endl;

  myUtils::addArray<double>(x_k,dualVar_,x_k,nDualVar_);

  myUtils::scaleVector<double>(-1,dualVar_,diffVec,nDualVar_);
  myUtils::addArray<double>(x_k,diffVec,diffVec,nDualVar_);

  double iterEnergy = computeEnergy(x_k);

  int lsCnt = 0;

  double rhsEnergy = curEnergy_ + myUtils::dotVec<double>(gradient_,diffVec,nDualVar_) + (L/2)*myUtils::dotVec<double>(diffVec,diffVec,nDualVar_);

  while (iterEnergy > rhsEnergy) {
   ++lsCnt;

   L *= beta;

   invL = -1./L;

   myUtils::scaleVector<double>(invL,gradient_,x_k,nDualVar_);
   myUtils::addArray<double>(x_k,dualVar_,x_k,nDualVar_);

   myUtils::scaleVector<double>(-1,dualVar_,diffVec,nDualVar_);
   myUtils::addArray<double>(x_k,diffVec,diffVec,nDualVar_);

   iterEnergy = computeEnergy(x_k);

   rhsEnergy = curEnergy_ + myUtils::dotVec<double>(gradient_,diffVec,nDualVar_) + (L/2.)*myUtils::dotVec<double>(diffVec,diffVec,nDualVar_);

   //std::cout<<"Iter energy "<<iterEnergy<<" Rhs energy "<<rhsEnergy<<" L "<<L<<std::endl;
  }

  std::cout<<"no. of line search iterations "<<lsCnt<<std::endl;

  t_k = (1 + sqrt(1 + 4*pow(t_prev,2)))/2;

  myUtils::scaleVector<double>(-1,x_prev,x_prev,nDualVar_);
  myUtils::addArray<double>(x_k,x_prev,x_prev,nDualVar_);
  myUtils::scaleVector<double>((t_prev-1)/t_k,x_prev,x_prev,nDualVar_);

  myUtils::addArray<double>(x_k,x_prev,dualVar_,nDualVar_);

  distributeDualVars();

  x_prev = x_k;

  t_prev = t_k;

  std::cout<<"FISTA L: "<<L<<std::endl;

  double curEnergyTime = 0;

  if (cntIter%10 == 0) {
   std::cout<<"gradient norm "<<gradNorm<<" gradient max "<<gradMax<<" Energy "<<curEnergy_<<std::endl;
   double tEnergy = myUtils::getTime();

   recoverFracPrimal();
   recoverFeasPrimal();
   recoverMaxPrimal(primalConsist_);

   std::cout<<std::fixed;
   std::cout<<std::setprecision(6);
   std::cout<<"Fractional primal energy: "<<compSmoothPrimalEnergy()<<" Smooth dual energy: "<<curEnergy_<<" Integral primal energy: "<<compIntPrimalEnergy();

   recoverMaxPrimal(primalFrac_);
   std::cout<<" Integral primal energy (directly from dual): "<<compIntPrimalEnergy()<<" Non-smooth dual energy: "<<computeNonsmoothDualEnergy()<<std::endl;

   totEnergyTime += myUtils::getTime() - tEnergy;
   curEnergyTime = myUtils::getTime() - tEnergy;
  }

  std::cout<<"ITERATION "<<cntIter<<" took "<<(myUtils::getTime() - tFull - curEnergyTime)<<" seconds. ITERATION "<<cntIter+1<<" starts, with tau "<<tau_<<"."<<std::endl; //" "<<computeNonsmoothDualEnergy()<<" "<<compIntPrimalEnergy()<<std::endl;
 }

 recoverFracPrimal();
 recoverFeasPrimal();
 recoverMaxPrimal(primalConsist_);

 std::cout<<std::fixed;
 std::cout<<std::setprecision(6);
 std::cout<<"solveFista: Fractional primal energy: "<<compSmoothPrimalEnergy()<<std::endl;
 std::cout<<"solveFista: Smooth dual energy: "<<computeEnergy(dualVar_)<<std::endl;
 std::cout<<"solveFista: Integral primal energy (feasible primal): "<<compIntPrimalEnergy()<<std::endl; 
 recoverMaxPrimal(primalFrac_);
 std::cout<<" Integral primal energy (directly from dual): "<<compIntPrimalEnergy();

 std::cout<<"solveFista: Non-smooth dual energy: "<<computeNonsmoothDualEnergy()<<std::endl;

 std::cout<<"Total time spent on computing energies: "<<totEnergyTime<<std::endl;

 return 0;
}

int dualSys::solveTrueLM()
{
 double tCompEnergies = 0;

 bool contIter = true;

 int cntIter = 0, cntIterTau = 1;

 double gradBasedDamp;
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
   popGradHessEnergy();
  }
  else {
   for (int i = 0 ; i != nDualVar_; ++i) {
    eigenHess_.coeffRef(i,i) += dampLambda_ - prevLambda;
   }
  }

  std::cout<<"solveNewton: populated gradient, hessian and energy "<<(myUtils::getTime() - tGHE)<<" seconds."<<std::endl;

  double gradNorm = myUtils::norm<double>(gradient_, nDualVar_);

  if (gradDampFlag) {
   gradBasedDamp = gradNorm/6;
   gradDampFlag = false;
  }

  if ((annealIval_ != -1) && (gradNorm < gradBasedDamp) && (tau_ < tauMax_)) {
   tau_ *= 2;
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

  if ((tau_ < tauMax_) || (gradMax > gradTol_) && (gradNorm > gradTol_)) {
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

   double nxtEnergy = computeEnergy(z);

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
  std::cout<<" Integral primal energy (directly from dual): "<<compIntPrimalEnergy()<<" Non-smooth dual energy: "<<computeNonsmoothDualEnergy()<<std::endl;

  std::cout<<"Computing energies took "<<myUtils::getTime() - tEnergy<<" seconds."<<std::endl;
  std::cout<<"NOTE: ITERATION "<<cntIter<<" took "<<(tEnergy - tFull)<<" seconds. The following figure includes unnecessary energy computation."<<std::endl;

  tCompEnergies += myUtils::getTime() - tEnergy;
#endif

  std::cout<<"ITERATION "<<cntIter<<" took "<<(myUtils::getTime() - tFull)<<" seconds. ITERATION "<<cntIter+1<<" starts, with tau "<<tau_<<"."<<std::endl; //" "<<computeNonsmoothDualEnergy()<<" "<<compIntPrimalEnergy()<<std::endl;
 }

 recoverFracPrimal();
 recoverFeasPrimal();
 recoverMaxPrimal(primalConsist_);

 std::cout<<"solveNewton: Fractional primal energy: "<<compSmoothPrimalEnergy()<<std::endl;
 std::cout<<"solveNewton: Smooth dual energy: "<<computeEnergy(dualVar_)<<std::endl;
 std::cout<<"solveNewton: Integral primal energy (feasible): "<<compIntPrimalEnergy()<<std::endl;

 recoverMaxPrimal(primalFrac_);
 std::cout<<"solveNewton: Integral primal energy (directly from dual): "<<compIntPrimalEnergy()<<std::endl;
 std::cout<<"solveNewton: Non-smooth dual energy: "<<computeNonsmoothDualEnergy()<<std::endl;

 std::cout<<"Total time spent on computing energies in each iteration: "<<tCompEnergies<<std::endl;

 return 0;
}
