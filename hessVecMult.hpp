#include <vector>
#include "dualSys.hpp"
#include "subProblem.hpp"

int matVecMult(const dualSys &myDual, const std::vector<double> &multVec, std::vector<double> &retVec) {

 int nNode = myDual.nNode_;
 int numLabTot = myDual.numLabTot_;
 int nCliq = myDual.nCliq_;
 double tau = myDual.tau_;
 std::vector<short> nLabelVec = myDual.nLabel_;
 std::vector<subProblem> subProb = myDual.subProb_;
 std::vector<std::vector<int> > cliqPerNode = myDual.cliqPerNode_;
 std::vector<double> uEnergy = myDual.uEnergy_;
 std::vector<int> unaryOffset = myDual.unaryOffset_;
 double dampLambda = myDual.dampLambda_;
 int nDualVar = myDual.nDualVar_;

 double f2DenExpMax = 0;
 std::vector<double> f2Den(nNode);
 std::vector<double> f2DenWt(nNode);
 std::vector<double> f2LabelSumWt(numLabTot);
 std::vector<double> f2LabelSum(numLabTot);

 for (int curNode = 0; curNode < nNode; ++curNode) {
  double expVal;

  int nLabel = nLabelVec[curNode];

  std::vector<double> f2DenExp;

  std::vector<double> multVecSum(nLabel);

  for (int i = 0; i != nLabel; ++i) {
   double cliqSumLabel = 0;

   for (std::vector<int>::iterator j = cliqPerNode[curNode].begin(); j != cliqPerNode[curNode].end(); ++j) {
    int nodeInd = myUtils::findNodeIndex(subProb[*j].memNode_, curNode);

    cliqSumLabel += subProb[*j].getDualVar()[subProb[*j].getNodeOffset()[nodeInd] + i];
    multVecSum[i] += multVec[subProb[*j].getCliqOffset() + subProb[*j].getNodeOffset()[nodeInd] + i];
   }
   f2DenExp.push_back(tau*(uEnergy[unaryOffset[curNode] + i] + cliqSumLabel));
  }

  f2DenExpMax = *std::max_element(f2DenExp.begin(), f2DenExp.end());

  for (int i = 0; i != nLabel; ++i) {
   expVal = exp(f2DenExp[i] - f2DenExpMax);
   myUtils::checkRangeError(expVal);
   double expValWt = expVal*multVecSum[i]*tau;

   f2LabelSumWt[unaryOffset[curNode] + i] = expValWt;
   f2LabelSum[unaryOffset[curNode] + i] = expVal;
   
   f2Den[curNode] += expVal;
   f2DenWt[curNode] += expValWt;
  }
 }

 for (int cliq = 0; cliq < nCliq; ++cliq) {
  int nCliqLab = subProb[cliq].nCliqLab_;
  int sizCliq = subProb[cliq].sizCliq_;
  int cliqOffset = subProb[cliq].getCliqOffset();
  std::vector<short> nLabel = subProb[cliq].getNodeLabel();
  std::vector<double> dualVar = subProb[cliq].getDualVar();
  std::vector<double> cEnergy = subProb[cliq].getCE();
  std::vector<int> nodeOffset = subProb[cliq].getNodeOffset();

  std::vector<double> nodeLabSumWt(dualVar.size());
  std::vector<double> nodeLabSum(dualVar.size());

  double expVal;
  double f1DenExpMax = 0;
  double f1DenWt = 0, f1Den = 0;

  std::vector<double> f1NodeSum;
  std::vector<double> f1DenExp;
 
  for (int i = 0; i != nCliqLab; ++i) {
   double nodeSum = 0;

   double labPull = nCliqLab;

   for (int j = 0; j != sizCliq; ++j) {
    labPull /= nLabel[j];

    int labPullTwo = ceil((i+1)/labPull);

//    if ((i+1) % labPull == 0) {
//    labPullTwo = (i+1)/labPull;
//    }
//    else {
//     labPullTwo = ((i+1)/labPull) + 1;
//    }

    int cliqLab;

    if (labPullTwo % nLabel[j] == 0) {
     cliqLab = nLabel[j] - 1;
    }
    else {
     cliqLab = (labPullTwo % nLabel[j]) - 1;
    }

    nodeSum += dualVar[nodeOffset[j] + cliqLab];
   }

   f1NodeSum.push_back(nodeSum);
   f1DenExp.push_back(tau*(cEnergy[i] - nodeSum));
  }

  f1DenExpMax = *std::max_element(f1DenExp.begin(), f1DenExp.end());

  for (int i = 0; i != nCliqLab; ++i) {
   expVal = exp(f1DenExp[i] - f1DenExpMax);
   myUtils::checkRangeError(expVal);

   f1Den += expVal;

   std::vector<int> cliqLab;

   double multVecSum = 0;

   expVal = exp(tau*(cEnergy[i] - f1NodeSum[i]) - f1DenExpMax);
   myUtils::checkRangeError(expVal);

   double labPull = nCliqLab;

   for (int j = 0; j != sizCliq; ++j) {
    labPull /= nLabel[j];

    int labPullTwo = ceil((i+1)/labPull);

//    if ((i+1) % labPull == 0) {
//    labPullTwo = (i+1)/labPull;
//    }
//    else {
//     labPullTwo = ((i+1)/labPull) + 1;
//    }

    if (labPullTwo % nLabel[j] == 0) {
     cliqLab.push_back(nLabel[j] - 1);
    }
    else {
     cliqLab.push_back((labPullTwo % nLabel[j]) - 1);
    }

    multVecSum +=  multVec[cliqOffset + nodeOffset[j] + cliqLab[j]]; 

    nodeLabSum[nodeOffset[j] + cliqLab[j]] += expVal;
   }

   double expValWt = expVal*multVecSum*tau;
   f1DenWt += expValWt;

   for (int j = 0; j != sizCliq; ++j) {
    nodeLabSumWt[nodeOffset[j] + cliqLab[j]] += expValWt;
   }
  }

  f1DenWt /= f1Den;

  for (int j = 0; j != sizCliq; ++j) {
   int curNode = subProb[cliq].memNode_[j];
   for (int k = 0; k != nLabel[j]; ++k) {
    retVec[cliqOffset + nodeOffset[j] + k] = (-1*nodeLabSum[nodeOffset[j] + k]*f1DenWt + nodeLabSumWt[nodeOffset[j] + k])/f1Den + \
                                             (-1*(f2LabelSum[unaryOffset[curNode] + k]/f2Den[curNode])*(f2DenWt[curNode]/f2Den[curNode]) + (f2LabelSumWt[unaryOffset[curNode] + k]/f2Den[curNode]));
//    retVec[cliqOffset + nodeOffset[j] + k] = -1*(f2LabelSum[unaryOffset[curNode] + k]/f2Den[curNode])*(f2DenWt[curNode]/f2Den[curNode]) + (f2LabelSumWt[unaryOffset[curNode] + k]/f2Den[curNode]);
//    retVec[cliqOffset + nodeOffset[j] + k] = (-1*nodeLabSum[nodeOffset[j] + k]*f1DenWt + nodeLabSumWt[nodeOffset[j] + k])/f1Den;
   }
  }
 }

 for (int dampIter = 0; dampIter != nDualVar; ++dampIter) {
  retVec[dampIter] += dampLambda*multVec[dampIter];
 }

 return 0;
}
