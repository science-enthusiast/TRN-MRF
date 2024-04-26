#ifndef SUBPROBLEM_HPP
#define SUBPROBLEM_HPP

#include <vector>
#include <set>

class subProblem
{
 std::vector<double> *pCEnergy_;
 std::vector<std::pair<std::vector<short>,double> > *pSpCEnergy_;
 std::vector<short> nLabel_;
 short comLabel_;
 int blkSiz_;
 
 std::vector<int> nodeOffset_;
 int nDualVar_;
 bool sparseFlag_;

public:
 subProblem(const std::vector<int>& memNode, std::vector<short> nLabel, std::vector<double>* pCEnergy): pCEnergy_(pCEnergy), nLabel_(nLabel), nDualVar_(0), sizCliq_(memNode.size()), nCliqLab_(1), memNode_(memNode) {
  sparseFlag_ = false;
  
  stride_.resize(nLabel.size());
  int strideComp = 1;
  int strideInd = nLabel.size() - 1;

  for (std::vector<short>::reverse_iterator rl = nLabel.rbegin(); rl != nLabel.rend(); ++rl) {
   stride_[strideInd] = strideComp;
   strideComp *= *rl;
   --strideInd;
  }

  for (std::vector<short>::iterator l = nLabel.begin(); l != nLabel.end(); ++l) {
   nodeOffset_.push_back(nDualVar_);
   nDualVar_ += *l;
   nCliqLab_ *= *l;
  }

  dualVar_.resize(nDualVar_);
  momentum_.resize(nDualVar_);
  gradient_.resize(nDualVar_);
  newtonStep_.resize(nDualVar_);
  primalCliqFrac_.resize(nCliqLab_);
  primalCliqConsist_.resize(nCliqLab_);
 }

 subProblem(const std::vector<int>& memNode, std::vector<short> nLabel, std::vector<std::pair<std::vector<short>,double> >* pSpCEnergy): pSpCEnergy_(pSpCEnergy), nLabel_(nLabel), nDualVar_(0), sizCliq_(memNode.size()), nCliqLab_(1), memNode_(memNode) {
  sparseFlag_ = true;
  
  stride_.resize(nLabel.size());
  int strideComp = 1;
  int strideInd = nLabel.size() - 1;

  for (std::vector<short>::reverse_iterator rl = nLabel.rbegin(); rl != nLabel.rend(); ++rl) {
   stride_[strideInd] = strideComp;
   strideComp *= *rl;
   --strideInd;
  }

  for (std::vector<short>::iterator l = nLabel.begin(); l != nLabel.end(); ++l) {
   nodeOffset_.push_back(nDualVar_);
   nDualVar_ += *l;
  }

  nCliqLab_ = (*pSpCEnergy).size();

  dualVar_.resize(nDualVar_);
  momentum_.resize(nDualVar_);
  gradient_.resize(nDualVar_);
  newtonStep_.resize(nDualVar_);
  primalCliqFrac_.resize(nCliqLab_);
  primalCliqConsist_.resize(nCliqLab_);
 }

 subProblem(const std::vector<int>& memNode, std::vector<short> nLabel, int rowSiz, int colSiz, double *sparseKappa, std::map<int,double> *sparseEnergy, std::set<int> *sparseInd): nLabel_(nLabel), nDualVar_(0), sizCliq_(memNode.size()), rowSiz_(rowSiz), colSiz_(colSiz), nCliqLab_(1), memNode_(memNode), sparseKappa_(sparseKappa), sparseEnergy_(sparseEnergy), sparseInd_(sparseInd) {
  int nMargSet = 6; //needed for calculating node-pair marginals

  sparseFlag_ = true;

  comLabel_ = nLabel_[0]; //if all nodes share same number of labels

  blkSiz_ = comLabel_*comLabel_;

  stride_.resize(sizCliq_);
  int strideComp = 1;
  int strideInd = sizCliq_ - 1;

  for (std::vector<short>::reverse_iterator rl = nLabel.rbegin(); rl != nLabel.rend(); ++rl) {
   stride_[strideInd] = strideComp;
   strideComp *= *rl;
   --strideInd;
  }

  for (std::vector<short>::iterator l = nLabel.begin(); l != nLabel.end(); ++l) {
   nodeOffset_.push_back(nDualVar_);
   nDualVar_ += *l;
   nCliqLab_ *= *l;
  }

  for (int iNode = 0; iNode != (sizCliq_+1)/2; ++iNode) {
   oneSet_.push_back(iNode);
  }

  for (int iNode = (sizCliq_+1)/2; iNode != sizCliq_; ++iNode) {
   twoSet_.push_back(iNode);
  }

  if (sizCliq_ == 3) { //only possible if clique is 1X3 or 3X1
   threeASet_.push_back(0);
   threeBSet_.push_back(2);
   fourASet_.push_back(1);
   fourBSet_.clear();

   fiveASet_.push_back(1);
   fiveBSet_.push_back(2);
   sixASet_.push_back(0);
   sixBSet_.clear();
  }
  else if ((rowSiz_ == 2) && (colSiz_ == 2)) {
   threeASet_.push_back(0);
   threeBSet_.push_back(2);

   fourASet_.push_back(1);
   fourBSet_.push_back(3);

   fiveASet_.push_back(0);
   fiveBSet_.push_back(3);

   sixASet_.push_back(1);
   sixBSet_.push_back(2);
  }

  setStride_.resize(nMargSet);
  nSetLab_.resize(nMargSet);

  for (int setInd = 0; setInd != nMargSet; ++setInd) {

   std::vector<int> fullSet = getSet(setInd);

   setStride_[setInd].resize(fullSet.size());
   strideInd = setStride_[setInd].size() - 1;
   strideComp = 1;

   nSetLab_[setInd] = 1;

   for (std::vector<int>::reverse_iterator rn = fullSet.rbegin(); rn != fullSet.rend(); ++rn) {
    setStride_[setInd][strideInd] = strideComp;
    strideComp *= nLabel[*rn];
    nSetLab_[setInd] *= nLabel[*rn];
    --strideInd;
   }
  }

  dualVar_.resize(nDualVar_);
  momentum_.resize(nDualVar_);
  gradient_.resize(nDualVar_);
  newtonStep_.resize(nDualVar_);
  //primalCliqFrac_.resize(nCliqLab_);
  //primalCliqConsist_.resize(nCliqLab_);
 }

 subProblem(const std::vector<int>& memNode, std::vector<short> nLabel, std::vector<double> pixVal): nLabel_(nLabel), nDualVar_(0), sizCliq_(memNode.size()), nCliqLab_(1), memNode_(memNode), pixVal_(pixVal) {

  stride_.resize(nLabel.size());
  int strideComp = 1;
  int strideInd = nLabel.size() - 1;

  for (std::vector<short>::reverse_iterator rl = nLabel.rbegin(); rl != nLabel.rend(); ++rl) {
   stride_[strideInd] = strideComp;
   strideComp *= *rl;
   --strideInd;
  }

  for (std::vector<short>::iterator l = nLabel.begin(); l != nLabel.end(); ++l) {
   nodeOffset_.push_back(nDualVar_);
   nDualVar_ += *l;
   nCliqLab_ *= *l;
  }

  dualVar_.resize(nDualVar_);
  momentum_.resize(nDualVar_);
  gradient_.resize(nDualVar_);
  newtonStep_.resize(nDualVar_);
  primalCliqFrac_.resize(nCliqLab_);
  primalCliqConsist_.resize(nCliqLab_);
 }

 int sizCliq_;
 int rowSiz_, colSiz_;
 int nCliqLab_;
 std::vector<int> nSetLab_;
 int vecOffset_;

 std::vector<double> dualVar_;
 std::vector<double> momentum_;
 std::vector<double> gradient_;
 std::vector<double> newtonStep_;

 std::vector<int> memNode_;
 std::vector<double> pixVal_;
 std::vector<int> oneSet_, twoSet_;

 std::vector<int> threeASet_, threeBSet_;
 std::vector<int> fourASet_, fourBSet_;
 std::vector<int> fiveASet_, fiveBSet_;
 std::vector<int> sixASet_, sixBSet_;

 std::set<int> cliqNeigh_;

 std::vector<int> stride_;
 std::vector<std::vector<int> > setStride_;

 int primalCliqMax_;
 std::vector<double> primalCliqFrac_;
 std::vector<double> primalCliqConsist_;

 std::vector<double> nodeLabSum_;
 std::vector<double> nodePairLabSum_;
 double f1Den_;

 std::vector<short> getNodeLabel() const {return nLabel_;}
 short getComLabel() const {return comLabel_;}
 int getBlkSiz() const {return blkSiz_;}
 int getCliqSiz() const {return sizCliq_;}
 std::vector<double> getDualVar() const {return dualVar_;}
 std::vector<double> getMomentum() const {return momentum_;}
 std::vector<int> getNodeOffset() const {return nodeOffset_;}
 int getCliqOffset() const {return vecOffset_;}
 void setCliqOffset(int vecOffset) {vecOffset_ = vecOffset;}
 int getDualSiz() const {return nDualVar_;}
 std::vector<double> getCE() const {return *pCEnergy_;}
 std::vector<double> getPixVal() const {return pixVal_;}

 std::vector<int> getMemNode() const {return memNode_;}
 //std::vector<int> getOneSet() const {return oneSet_;}
 //std::vector<int> getTwoSet() const {return twoSet_;}

 std::vector<int> getSet(int setInd) const {
 
  std::vector<int> retVec;

  switch(setInd) {
   case 0:
    return oneSet_;
   case 1:
    return twoSet_;
   case 2:
    retVec = threeASet_;
    retVec.insert(retVec.end(), threeBSet_.begin(), threeBSet_.end());

    return retVec;
   case 3:
    retVec = fourASet_;
    retVec.insert(retVec.end(), fourBSet_.begin(), fourBSet_.end());

    return retVec;
   case 4:
    retVec = fiveASet_;
    retVec.insert(retVec.end(), fiveBSet_.begin(), fiveBSet_.end());

    return retVec;
   case 5:
    retVec = sixASet_;
    retVec.insert(retVec.end(), sixBSet_.begin(), sixBSet_.end());

    return retVec;
  }

  return retVec;
 }

 std::vector<int> getSet(int setInd, char partInd) const {
  if (setInd == 2) {
   if (partInd == 'a') {
    return threeASet_;
   }
   else if (partInd == 'b') {
    return threeBSet_;
   }
  }
  else if (setInd == 3) {
   if (partInd == 'a') {
    return fourASet_;
   }
   else if (partInd == 'b') {
    return fourBSet_;
   }
  }
  else if (setInd == 4) {
   if (partInd == 'a') {
    return fiveASet_;
   }
   else if (partInd == 'b') {
    return fiveBSet_;
   }
  }
  else if (setInd == 5) {
   if (partInd == 'a') {
    return sixASet_;
   }
   else if (partInd == 'b') {
    return sixBSet_;
   }
  }
 }

 std::vector<int> getStride() const {return stride_;}

 std::vector<int> getStride(int setInd) const {
  return setStride_[setInd];
 }

 int getSetLabCnt(int setInd) const {
  return nSetLab_[setInd];
 }

 std::set<int> getSparseInd() const {return *sparseInd_;}

 double getCEConst() const {return *sparseKappa_;}

 double getCE(int iCliqLab) const {
  if (sparseFlag_) {
   if ((*sparseEnergy_).find(iCliqLab) == (*sparseEnergy_).end()) {
    return *sparseKappa_;
   }
   else {
    return (*sparseEnergy_)[iCliqLab];
   }
  }
  else {
    return (*pCEnergy_)[iCliqLab];
  }
 }

// std::vector<std::pair<std::vector<short>,double> > getSparseCEnergy() const {return *pSpCEnergy_;}

 double *sparseKappa_;
 std::map<int,double> *sparseEnergy_;
 std::set<int> *sparseInd_;
};

#endif //SUBPROBLEM_HPP
