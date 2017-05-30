#include <string>
#include <cstring>
#include <map>
#include <vector>
#include <cmath>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

std::vector<double> birchTomasDisp(const int, const int, const int, const int, const int, const std::vector<double> &, const std::vector<double> &);

std::vector<double> popl1DispUEnergy(int rPos, int cPos, int nRow, int nCol, int nLabel, std::vector<double> lImg, std::vector<double> rImg)
{
  std::vector<double> uEnergy;

  int iL = 0;

  while ((iL < nLabel) && ((cPos + iL) < nCol)) {
   double rVal;

   rVal = rImg[rPos*nCol + cPos];

   double lVal;

   if (cPos + iL < 0) {
    lVal = lImg[rPos*nCol + cPos];
   }
   else {
    lVal = lImg[rPos*nCol + cPos + iL];
   }

   uEnergy.push_back(std::abs(rVal-lVal));

   ++iL;
  }

  int extraSz = nLabel - uEnergy.size();

  if (extraSz != 0) {
   double dataBuff = uEnergy.back();
   for (int i = 0; i != extraSz; ++i) {
    uEnergy.push_back(dataBuff);
   }
  }

  return uEnergy;
}

std::vector<double> popPropl1DispUEnergy(int rPos, int cPos, int nRow, int nCol, int nLabel, std::vector<double> lImg, std::vector<double> rImg)
{
  std::vector<double> uEnergy;

  int iL = 0;

  while ((iL < nLabel) && ((cPos + iL) < nCol)) {
   double rVal;

   rVal = rImg[rPos*nCol + cPos];

   double lVal;

   if (cPos + iL < 0) {
    lVal = lImg[rPos*nCol + cPos];
   }
   else {
    lVal = lImg[rPos*nCol + cPos + iL];
   }

   uEnergy.push_back(std::abs(rVal-lVal));

   ++iL;
  }

  int extraSz = nLabel - uEnergy.size();

  if (extraSz != 0) {
   double dataBuff = uEnergy.back();
   for (int i = 0; i != extraSz; ++i) {
    uEnergy.push_back(dataBuff);
   }
  }

  return uEnergy;
}

int popStereoSmoothEnergies(int nRow, int nCol, const std::vector<double> &lImg, const std::vector<double> &rImg, double gtScale, double downScale, std::vector<std::vector<double> > &uEnergies, double &sparseKappa, std::vector<std::map<int,double> >  &sparseEnergies, std::vector<std::vector<int> > &cliqNodes)
{
 std::ifstream propFilesList("proposalFiles.txt");

 std::vector<std::vector<double> > proposals;

 std::string propFile;

 while (propFilesList>>propFile) {
  std::cout<<propFile<<std::endl;

  std::ifstream propStream(propFile.c_str());

  double propVal;

  std::vector<double> propValVec;
  std::string propCurLine;

  while (propStream>>propCurLine) {
   std::stringstream propCurStream(propCurLine);
   while (propCurStream>>propVal) {
    propValVec.push_back(propVal);
   }
  }

  proposals.push_back(propValVec);
 }

 int nLabel = proposals.size();

 std::cout<<"number of proposals is "<<nLabel<<std::endl;

 double kappa = 2;
 double alphaMax = 1;

 sparseKappa = -1*kappa;

 int maxCliqSiz = 0, minCliqSiz = 4096;

 for (int x = 0; x != nCol; ++x) {
  for (int y = 0; y != nRow; ++y) {
   std::vector<int> curVec;

   int pixPos = y*nCol + x;
   //std::cout<<"pixel position "<<pixPos<<" ";

   if (((y == 0) || (y == nRow-1)) && (x != 0) && (x != nCol-1)) {
    curVec.push_back(pixPos-1);
    curVec.push_back(pixPos);
    curVec.push_back(pixPos+1);
    cliqNodes.push_back(curVec);
   }
   else if (((x == 0) || (x == nCol-1)) && (y != 0) && (y != nRow-1)) {
    curVec.push_back(pixPos-nCol);
    curVec.push_back(pixPos);
    curVec.push_back(pixPos+nCol);
    cliqNodes.push_back(curVec);
   }
   else if ((x != 0) && (y != 0)) {
    curVec.push_back(pixPos-1);
    curVec.push_back(pixPos);
    curVec.push_back(pixPos+1);
    cliqNodes.push_back(curVec);

    curVec.clear();

    curVec.push_back(pixPos-nCol);
    curVec.push_back(pixPos);
    curVec.push_back(pixPos+nCol);
    cliqNodes.push_back(curVec);
   }

   std::vector<double> curUEnergy;

   for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
    int disp = proposals[iLabel][pixPos]*gtScale*downScale;

    int rX = x - disp;

    int rPixPos = 0;

    if (rX < 0) {
     rPixPos = y*nCol;
    }
    else if (rX > nCol-1) {
     rPixPos = y*nCol + nCol-1;
    }
    else {
     rPixPos = y*nCol + rX;
    }

    curUEnergy.push_back(-1*std::abs(lImg[pixPos]-rImg[rPixPos]));
   }

   uEnergies.push_back(curUEnergy);
  }
 }

 for (std::vector<std::vector<int> >::iterator cliqIter = cliqNodes.begin(); cliqIter != cliqNodes.end(); ++cliqIter) {
  std::map<int,double> sparseEnergy;

  int sparseCnt = 0;

  int energyInd = 0;

  for (int i = 0; i != nLabel; ++i) {
   for (int j = 0; j != nLabel; ++j) {
    for (int k = 0; k != nLabel; ++k) {
     double Fx = proposals[i][(*cliqIter)[0]]-2*proposals[j][(*cliqIter)[1]]+proposals[k][(*cliqIter)[2]];

     double gradient = (proposals[k][(*cliqIter)[2]] - proposals[k][(*cliqIter)[0]])/2;

     if ((std::abs(Fx) <= kappa) && (std::abs(gradient) <= alphaMax)) {
      sparseEnergy[energyInd] = -1*Fx;
      ++sparseCnt;
     }
//     else {
//      cliqEnergy.push_back(-1*kappa);
//     }

     ++energyInd;
    }
   }
  }

  sparseEnergies.push_back(sparseEnergy);

  if (sparseCnt > maxCliqSiz) {
   maxCliqSiz = sparseCnt;
  }
  if (sparseCnt < minCliqSiz) {
   minCliqSiz = sparseCnt;
  }
 }

 std::cout<<"max. clique size "<<maxCliqSiz<<" min. clique size "<<minCliqSiz<<std::endl;

 return 0;
}

std::vector<double> popDenoiseUEnergy(int nLabel, double pixVal) {
 std::vector<double> uEnergy;

 for (int i = 0; i != nLabel; ++i) {
 //uEnergy.push_back(-1*pow(j-pixVal, 2));
  uEnergy.push_back(-1*std::abs(i-pixVal));
 }

 return uEnergy;
}

std::vector<double> popDenRelUEnergy(int nLabel) {
 std::vector<double> uEnergy;

 for (int i = -1*(nLabel/2); i < nLabel/2; ++i) {
 //uEnergy.push_back(-1*pow(j-pixVal, 2));
  uEnergy.push_back(-1*std::abs(i));
 }

 return uEnergy;
}

std::vector<double> popDenoiseCEnergy(int sizCliq, int nCliqLab, std::vector<std::vector<int> > cliqLab) {
 std::vector<double> cEnergy;

 std::cout<<"denoise clique energy ";

 double minCliq = 0;

 for (int i = 0; i != nCliqLab; ++i) {
  double avg = 0;
  for (int j = 0; j != sizCliq; ++j) {
   avg += cliqLab[i][j];
  }

  avg /= sizCliq;

  double var = 0;
  for (int j = 0; j != sizCliq; ++j) {
   var += pow(cliqLab[i][j]-avg,2);
  }

  cEnergy.push_back(-1*(var/sizCliq));

  std::cout<<cEnergy.back()<<" ";

  if (cEnergy.back() < minCliq) {
   minCliq = cEnergy.back();
  }
 }

 std::cout<<std::endl;

 std::cout<<"min. clique energy "<<minCliq<<std::endl;

 return cEnergy;
}

int popSparseVarCEnergy(int sizCliq, std::vector<short> nLabel, int nCliqLab, double &sparseKappa, std::map<int,double> &sparseEnergies) {

 sparseKappa = 5;

 double minCliqEnergy = 0;

 for (int iCliqLab = 0; iCliqLab != nCliqLab; ++iCliqLab) {
  double avg = 0;

  std::vector<short> cliqLab;
  double labPull = nCliqLab;

  for (int k = 0; k != sizCliq; ++k) {
   labPull /= nLabel[k];

   int labPullTwo = ceil((iCliqLab+1)/labPull);

   if (labPullTwo % nLabel[k] == 0) {
    cliqLab.push_back(nLabel[k] - 1);
   }
   else {
    cliqLab.push_back((labPullTwo % nLabel[k]) - 1);
   }

   avg += cliqLab.back();
  }

  avg /= sizCliq;

  double var = 0;
  for (int k = 0; k != sizCliq; ++k) {
   var += pow(cliqLab[k]-avg,2);
  }

  if (var/sizCliq < sparseKappa) {
   sparseEnergies[iCliqLab] = -1*(var/sizCliq);

   if (sparseEnergies[iCliqLab] < minCliqEnergy) {
    minCliqEnergy = sparseEnergies[iCliqLab];
   }
  }
 }

 sparseKappa *= -1;

 std::cout<<"sparse energy count "<<sparseEnergies.size()<<std::endl;
 std::cout<<"min. sparse clique energy "<<minCliqEnergy<<std::endl;

 return 0;
}

int popSparseVarCEnergy(int sizCliq, std::vector<short> nLabel, std::vector<int> cliqNodes, const std::vector<double> &ipImg, int nCliqLab, double &sparseKappa, std::map<int,double> &sparseEnergies) {

 sparseKappa = 5;

 double minCliqEnergy = 0;

 for (int iCliqLab = 0; iCliqLab != nCliqLab; ++iCliqLab) {
  double avg = 0;

  std::vector<short> cliqLab;
  double labPull = nCliqLab;

  for (int k = 0; k != sizCliq; ++k) {
   labPull /= nLabel[k];

   int labPullTwo = ceil((iCliqLab+1)/labPull);

   if (labPullTwo % nLabel[k] == 0) {
    cliqLab.push_back(nLabel[k] - 1);
   }
   else {
    cliqLab.push_back((labPullTwo % nLabel[k]) - 1);
   }

   avg += cliqLab.back() - nLabel[k] + ipImg[cliqNodes[k]];
  }

  avg /= sizCliq;

  double var = 0;
  for (int k = 0; k != sizCliq; ++k) {
   double tempVal = cliqLab[k] - nLabel[k] + ipImg[cliqNodes[k]];

   var += pow(tempVal-avg,2);
  }

  if (var/sizCliq < sparseKappa) {
   sparseEnergies[iCliqLab] = -1*(var/sizCliq);

   if (sparseEnergies[iCliqLab] < minCliqEnergy) {
    minCliqEnergy = sparseEnergies[iCliqLab];
   }
  }
 }

 sparseKappa *= -1;

 std::cout<<"sparse energy count "<<sparseEnergies.size()<<std::endl;
 std::cout<<"min. sparse clique energy "<<minCliqEnergy<<std::endl;

 return 0;
}

std::vector<double> popSparseDenoiseCEnergy(int nLabel) {
 double alphaMax = 1;
 double kappa = 2; //*nLabel;

 int sparseCnt = 0;

 std::vector<double> cEnergy;

 for (int i = 0; i != nLabel; ++i) {
  for (int j = 0; j != nLabel; ++j) {
   for (int k = 0; k != nLabel; ++k) {
    int Fx = i-2*j+k;

    double grad = (k-i)/2.0;

    if ((std::abs(Fx) <= kappa) && (std::abs(grad) <= alphaMax)) {
     //cEnergy.push_back(-1*std::abs(Fx));
     cEnergy.push_back(0);
     ++sparseCnt;
     //std::cout<<"pattern is "<<i<<" "<<j<<" "<<k<<std::endl;
    }
    else {
     cEnergy.push_back(-1*kappa);
    }
   }
  }
 }

 std::cout<<"Total number of potentials "<<pow(nLabel,3)<<" sparse count is "<<sparseCnt<<std::endl;

 return cEnergy;
}

int popSparseDenoiseCEnergy(int nLabel, double &sparseKappa, std::map<int,double> &sparseEnergies) {
 double alphaMax = 1;
 sparseKappa = 1; //*nLabel;

 int sparseCnt = 0;

 for (int i = 0; i != nLabel; ++i) {
  for (int j = 0; j != nLabel; ++j) {
   for (int k = 0; k != nLabel; ++k) {
    int Fx = i-2*j+k;

    double grad = (k-i)/2.0;

    int sparseInd = i*pow(nLabel,2) + j*nLabel + k;

    if ((std::abs(Fx) <= sparseKappa) && (std::abs(grad) <= alphaMax)) {
     sparseEnergies[sparseInd] = 0;
     ++sparseCnt;
     //std::cout<<"pattern is "<<i<<" "<<j<<" "<<k<<std::endl;
    }
   }
  }
 }

 sparseKappa *= -1;

 return 0;
}

std::vector<double> popFoEUEnergy(int nLabel, double pixVal) {
 std::vector<double> uEnergy;

 double sigma = 4;

 for (int i = -1*nLabel/2; i != nLabel/2; ++i) {
  uEnergy.push_back(-1*pow(i,2)/(2*pow(sigma,2)));
 }

 return uEnergy;
}

std::vector<double> popFoECEnergy(int nLabel, std::vector<double> pixVals) {
 std::vector<double> cEnergy;

 int numFilter = 3;
 int patchSiz = 4;

 int totLabels = pow(nLabel,patchSiz); //assumes all pixels have same number of labels

 // FoE experts provided by Stefan Roth
 double alpha[] = {0.586612685392731, 1.157638405566669, 0.846059486257292};
 double expert[][4] = {
	{-0.0582774013402734, 0.0339010363051084, -0.0501593018104054, 0.0745568557931712},
	{0.0492112815304123, -0.0307820846538285, -0.123247230948424, 0.104812330861557},
	{0.0562633568728865, 0.0152832583489560, -0.0576215592718086, -0.0139673758425540}
 };

 std::vector<int> strideVec;
 int stridePow = patchSiz - 1;

 for (int i = 0; i != patchSiz; ++i) {
  strideVec.push_back(pow(nLabel,stridePow));
  --stridePow;
 }

 std::vector<int> labelOffset;

 for (int i = -1*(nLabel/2); i != nLabel/2; ++i) {
  labelOffset.push_back(i);
 }

 for (int i = 0; i != totLabels; ++i) {
  std::vector<int> assignLabel;

  for (int j = 0; j != patchSiz; ++j) {
   assignLabel.push_back(i/strideVec[j] % nLabel);
  }

  double curCEnergy = 0;

  for (int j = 0; j != numFilter; ++j) {
   double localSum = 0;
   for (int k = 0; k != patchSiz; ++k) {
    localSum += expert[j][k]*pixVals[k];
   }

   curCEnergy += alpha[j]*log(1 + 0.5* pow(localSum,2));
  }

  cEnergy.push_back(curCEnergy);
 }

 return cEnergy;
}

std::vector<double> popSqCEnergy(int nLabel) {
 double kappa = 10; //*nLabel;

 int sparseCnt = 0;

 std::vector<double> cEnergy;

 for (int i = 0; i != nLabel; ++i) {
  for (int j = 0; j != nLabel; ++j) {
   double sqEnergy = pow((i-j),2);

   if (sqEnergy <= kappa) {
    cEnergy.push_back(-1*sqEnergy);
    ++sparseCnt;
   }
   else {
    cEnergy.push_back(-1*kappa);
   }
  }
 }

 std::cout<<"Total number of potentials "<<pow(nLabel,3)<<" sparse count is "<<sparseCnt<<std::endl;

 return cEnergy;
}

std::vector<double> popUniformUEnergy(int nLabel) {
 std::vector<double> uEnergy(nLabel);

 for (int i = 0; i != nLabel; ++i) {
  double randVal = static_cast<double>(rand() % 1000)/1000;
  uEnergy[i] = -1*randVal;
 }

 return uEnergy;
}

std::vector<double> popUniformCEnergy(int nCliqLab) {
 std::vector<double> cEnergy(nCliqLab);

 for (int i = 0; i != nCliqLab; ++i) {
  double randVal = static_cast<double>(rand() % 1000)/1000;
  cEnergy[i] = -1*randVal;
 }

 return cEnergy;
}

int popSparseUniformCEnergy(int nCliqLab, double &sparseKappa, std::map<int,double> &sparseEnergies) {
 std::cout<<"Sparse Uniformly Random Clique Energy"<<std::endl;

 sparseKappa = 3;

 for (int iCliqLab = 0; iCliqLab != nCliqLab; ++iCliqLab) {
  double randVal = static_cast<double>(rand() % 100000)/100;

  if (randVal < sparseKappa) {
   sparseEnergies[iCliqLab] = -1*randVal*0.5;
  }
 }

 std::cout<<"Total number of energies "<<nCliqLab<<" sparse energies "<<sparseEnergies.size()<<std::endl;

 sparseKappa *= -0.5;

 return 0;
}

std::vector<double> popUniformCEnergy(std::vector<int> cliqNode, std::vector<short> nLabel) {
 int nCliqLab = 1;

 for (std::vector<int>::iterator iNode = cliqNode.begin(); iNode != cliqNode.end(); ++iNode) {
  nCliqLab *= nLabel[*iNode];
 }

 std::vector<double> cEnergy(nCliqLab);

 for (int i = 0; i != nCliqLab; ++i) {
  double randVal = static_cast<double>(rand() % 1000)/100;
  cEnergy[i] = -1*randVal;
 }

 return cEnergy;
}

std::vector<double> popStrongGraphMatchUEnergy(int nodeInd, int nLabel) {
 std::vector<double> uEnergy(nLabel);

 //int sqrtLab = static_cast<int>(sqrt(nLabel));

 //int nodeLab = nodeInd % sqrtLab;

 std::cout<<"uEnergy for node "<<nodeInd<<":";

 for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
  //int curLab = iLabel % sqrtLab;

  //uEnergy[iLabel] = -1*std::abs(static_cast<double>(nodeLab-curLab)/sqrtLab);
  uEnergy[iLabel] = -1*std::abs(static_cast<double>(iLabel-nodeInd)/nLabel);
  //uEnergy[iLabel] = 0;
  //uEnergy[iLabel] = -10*std::abs(static_cast<double>(iLabel-nodeInd));

  std::cout<<" "<<uEnergy[iLabel];
 }

 std::cout<<std::endl;

 return uEnergy;
}

std::vector<std::vector<double> > popSiftMatchUEnergy() {
 double uScale = 0;

 std::vector<std::vector<double> > uEnergyVec;

 std::vector<std::vector<double> > siftDescOne;
 std::vector<std::vector<double> > siftDescTwo;

 std::string curLine;

 std::ifstream siftFile("/home/hari/Dropbox/input/house/img1Sift64.txt");

 while (std::getline(siftFile,curLine)) {
  std::stringstream curStream(curLine);

  std::vector<double> curSiftVec;

  double curSiftVal;

  while (curStream>>curSiftVal) {
   curSiftVec.push_back(curSiftVal);
  }

  siftDescOne.push_back(curSiftVec);
 }

 siftFile.close();

 siftFile.open("/home/hari/Dropbox/input/house/img2Sift94.txt");

 while (std::getline(siftFile,curLine)) {
  std::stringstream curStream(curLine);

  std::vector<double> curSiftVec;

  double curSiftVal;

  while (curStream>>curSiftVal) {
   curSiftVec.push_back(curSiftVal);
  }

  siftDescTwo.push_back(curSiftVec);
 }

 siftFile.close();

 int nLabel = siftDescTwo.size();

 //int sqrtLab = static_cast<int>(sqrt(nLabel));

 for (std::size_t nodeInd = 0; nodeInd != siftDescOne.size(); ++nodeInd) {

  std::vector<double> curUEnergy(nLabel);

  std::cout<<"uEnergy for node "<<nodeInd<<":";

  double normVal = 0;

  for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
   double curSSD = 0;

   std::vector<double>::iterator iSrcVal = siftDescOne[nodeInd].begin();

   for (std::vector<double>::iterator iTgtVal = siftDescTwo[iLabel].begin(); iTgtVal != siftDescTwo[iLabel].end(); ++iTgtVal) {
    curSSD += pow((*iSrcVal - *iTgtVal),2);
   }

   curUEnergy[iLabel] =uScale*curSSD;

   normVal += curSSD;
  }

  for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
   curUEnergy[iLabel] /= -1*normVal;
  }

  //std::cout<<std::endl;

  uEnergyVec.push_back(curUEnergy);
 }

 return uEnergyVec;
}

int popSynth2DGraphMatchCEnergy(std::string fileNamePrefix, std::vector<short> nLabel, std::vector<std::vector<int> > cliqNodes, std::vector<double> &sparseKappa, std::vector<std::map<int,double> > &sparseEnergies) {

 double energyScale = 1.0;

 std::cout<<"Clique energy scale = "<<energyScale<<std::endl;

 std::string nnDataFile = "/home/hari/softwares/ann_1.1.2/ann_1.1.2/bin/" + fileNamePrefix + "nnop.txt";

 std::cout<<"Nearest neighbours file "<<nnDataFile<<std::endl;

 //std::string nnDataFile = "/home/hari/Dropbox/input/" + fileNamePrefix + "nnop.txt";

 std::ifstream ipFile(nnDataFile.c_str());

 std::string curLine;

 int nnCnt = 30000;
 int queryCnt = 0;

 std::string twoDPath = "/home/hari/softwares/2d_graph_match/";

 std::string quDataFileName = twoDPath + fileNamePrefix + "qu.txt";

 std::ifstream queryTupleFile(quDataFileName.c_str());

 std::vector<std::vector<double> > queryTuple;

 while (std::getline(queryTupleFile,curLine)) {
  std::stringstream curStream(curLine);

  std::vector<double> curSinVec;
  //curStream>>curSinVec[0];
  //curStream>>curSinVec[1];
  //curStream>>curSinVec[2];

  double curVal;

  while (curStream >> curVal) {
   curSinVec.push_back(curVal);
  }

  queryTuple.push_back(curSinVec);

  ++queryCnt;
 }

 queryTupleFile.close();

 std::string tgtDataFileName = twoDPath + fileNamePrefix + "tgt.txt";
 std::string tgtPtsFileName = twoDPath + fileNamePrefix + "ptstgt.txt";

 std::ifstream tgtTupleFile(tgtDataFileName.c_str());
 std::ifstream tgtPtsTupleFile(tgtPtsFileName.c_str());

 std::vector<std::vector<double> > tgtTuple;
 std::vector<std::vector<int> > tgtPtsTuple;

 while (std::getline(tgtTupleFile,curLine)) {
  std::stringstream curStream(curLine);

  std::vector<double> curSinVec;
  double curVal;

  while (curStream >> curVal) {
   curSinVec.push_back(curVal);
  }

  tgtTuple.push_back(curSinVec);
 }

 tgtTupleFile.close();

 while (std::getline(tgtPtsTupleFile,curLine)) {
  std::stringstream curStream(curLine);

  std::vector<int> curPtsVec;
  int curVal;

  while (curStream >> curVal) {
   curPtsVec.push_back(curVal);
  }

  tgtPtsTuple.push_back(curPtsVec);
 }

 tgtPtsTupleFile.close();

 int queryInd = 0;

 while (std::getline(ipFile,curLine)) {
  if (curLine.find("Query") != std::string::npos) {
   std::set<int> curIndices, nnIndices;

   std::string nnData;

   std::vector<int> curNodes = cliqNodes[queryInd];
   std::vector<short> curLabel;

   int sizCliq = curNodes.size();

   for (std::vector<int>::iterator iNode = curNodes.begin(); iNode != curNodes.end(); ++iNode) {
    curLabel.push_back(nLabel[*iNode]);
   }

   for (int iNN = 0; iNN != nnCnt; ++iNN) {
    std::getline(ipFile,nnData);
    int nnInd, sparseInd = 0;

    std::stringstream nnDataStream(nnData);

    nnDataStream>>nnInd;
    nnDataStream>>nnInd;

    int labStride = 1;

    for (int iNode = sizCliq-1; iNode != 0; --iNode) {
     labStride *= curLabel[iNode];
    }

    for (int iNode = 0; iNode != sizCliq; ++iNode) {
     sparseInd += labStride*tgtPtsTuple[nnInd][iNode];
     if (iNode != sizCliq-1) {
      labStride /= curLabel[iNode+1];
     }
    }

    curIndices.insert(sparseInd);
    nnIndices.insert(nnInd);
   }

   std::map<int,double> curEnergies;

   std::vector<double> curQuery = queryTuple[queryInd];

   double gamma = 0;

   std::vector<double> l2Vec;

   for (std::set<int>::iterator iNN = nnIndices.begin(); iNN != nnIndices.end(); ++iNN) {
    std::vector<double> curTgt = tgtTuple[*iNN];

    //double l2Dist = pow(curTgt[0]-curQuery[0],2) + pow(curTgt[1]-curQuery[1],2) + pow(curTgt[2]-curQuery[2],2);
    double l2Dist = 0;

    for (std::vector<double>::iterator iTgt = curTgt.begin(), iQuery = curQuery.begin(); iTgt != curTgt.end(); ++iTgt, ++iQuery) {
     l2Dist += pow(*iTgt - *iQuery,2);
    }

    gamma += l2Dist;

    l2Vec.push_back(l2Dist);
   }

   gamma /= curIndices.size();

   gamma = 1/gamma;

   int locInd = 0;

   double A = 1;

   double maxEnergy = A*exp(-1*gamma*l2Vec[0]); //exp(-1*gamma*l2Vec[0]);
   double minEnergy = maxEnergy;

   //std::cout<<"Clique energies";

   for (std::set<int>::iterator iCur = curIndices.begin(); iCur != curIndices.end(); ++iCur) {
    double curEnergyVal = A*exp(-1*gamma*l2Vec[locInd]);
    //double curEnergyVal = A*exp(-1*l2Vec[locInd]);

    //std::cout<<" "<<curEnergyVal;

    curEnergies[*iCur] = curEnergyVal*energyScale;

    if (curEnergyVal > maxEnergy) {
     maxEnergy = curEnergyVal;
    }

    if (curEnergyVal < minEnergy) {
     minEnergy = curEnergyVal;
    }

    //curEnergies[*iCur] = sparseKappa*exp(-1*l2Vec[locInd]);
    ++locInd;
   }

   //std::cout<<std::endl;

#if 0
   int totLab = tgtTuple.size();
   for (int iLab = 0; iLab != totLab; ++iLab) {
    if ((curIndices.find(iLab) == curIndices.end()) && (curIndices.size() != 40000)) {
     curIndices.insert(iLab);
     curEnergies[iLab] = minEnergy;
    }
   }
   sparseKappa.push_back(minEnergy/2);
#endif

   sparseKappa.push_back(minEnergy*energyScale);

   std::cout<<"max. energy "<<maxEnergy<<" min. energy "<<minEnergy<<std::endl;

   if (locInd != nnCnt) {
    std::cout<<"sparse count not tallying. Should be "<<nnCnt<<", it is "<<locInd<<std::endl;
   }

   sparseEnergies.push_back(curEnergies);

   ++queryInd;
  }
 }

 std::cout<<"Total number of sparse indices "<<sparseEnergies.size()<<std::endl;

 std::cout<<"Query debug count "<<queryInd<<" Query generated count "<<queryCnt<<std::endl;

 return 0;
}

int popHighStereoEnergies(int nRow, int nCol, int nLabel, const std::vector<double> &lImg, const std::vector<double> &rImg, double gtScale, double downScale, std::vector<std::vector<double> > &uEnergies, const std::vector<std::vector<int> > &cliqNodes, std::vector<double> &sparseKappa, std::vector<std::map<int,double> > &sparseEnergies) {
 std::ifstream propFilesList("proposalFiles.txt");

 //all nodes are assumed to have the same label

 std::cout<<"number of proposals is "<<nLabel<<std::endl;

 std::vector<std::vector<double> > proposals(nLabel); //as many proposals as labels

 std::string propFile;

 for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
  propFilesList>>propFile;
  std::cout<<propFile<<std::endl;

  std::vector<double> propValVec;

#if 1
  std::ifstream propStream(propFile);

  double propVal;

  std::string propCurLine;

  while (propStream>>propCurLine) {
   std::stringstream propCurStream(propCurLine);
   while (propCurStream>>propVal) {
    if (propVal < 0) {
     propValVec.push_back(0);
    }
    else {
     propValVec.push_back(propVal);
    }
   }
  }
#else //for tsukuba
  for (int pixPos = 0; pixPos != nRow*nCol; ++pixPos) {
   propValVec.push_back(iLabel);
  }
#endif

  std::cout<<"proposal "<<iLabel<<" is of size "<<propValVec.size()<<std::endl;

  proposals[iLabel] = propValVec;
 }

 for (int y = 0; y != nRow; ++y) {
  for (int x = 0; x != nCol; ++x) {

   int pixPos = y*nCol + x;
   //std::cout<<"pixel position "<<pixPos<<" ";

   std::vector<double> curUEnergy(nLabel);

#if 1
   for (int iLabel = 0; iLabel != nLabel; ++iLabel) {
    int disp = proposals[iLabel][pixPos]*gtScale*downScale;

    int rX = x - disp;

    int rPixPos = 0;

    if (rX < 0) {
     rPixPos = y*nCol;
    }
    else if (rX > nCol-1) {
     rPixPos = y*nCol + nCol-1;
    }
    else {
     rPixPos = y*nCol + rX;
    }

    curUEnergy[iLabel] = -1*std::abs(lImg[pixPos]-rImg[rPixPos]);
   }
#else
   curUEnergy = birchTomasDisp(y, x, nRow, nCol, nLabel, lImg, rImg);
#endif

   uEnergies.push_back(curUEnergy);
  }
 }

 double kappa = 5;
 double alphaMax = 3;
 double FxMax = 0;
 bool wtFlag = false;
 double wtCutOff = 8;

 int cliqInd = 0;

 //std::cout<<"Sparse energies are ";

 int numCliq = cliqNodes.size();

 sparseKappa.resize(numCliq);
 sparseEnergies.resize(numCliq);

 int minSparseSiz = 100000;
 int maxSparseSiz = 0;

 for (std::vector<std::vector<int> >::const_iterator iCliq = cliqNodes.begin(); iCliq != cliqNodes.end(); ++iCliq) {
  int sparseCnt = 0;
  int zeroCnt = 0;
  int energyInd = 0;

  std::vector<int> nodes = *iCliq;

  double intGrad = std::abs((lImg[nodes[2]]-lImg[nodes[0]])/2);
  double energyWt = 1;

  if ((wtFlag) && (intGrad <= wtCutOff)) {
   energyWt = 2;
  }

  sparseKappa[cliqInd] = -1*energyWt*kappa;

  for (int i = 0; i != nLabel; ++i) {
   for (int j = 0; j != nLabel; ++j) {
    for (int k = 0; k != nLabel; ++k) {
     double Fx = proposals[i][nodes[0]]-2*proposals[j][nodes[1]]+proposals[k][nodes[2]];

     if (Fx > FxMax) {
      FxMax = Fx;
     }

     double depGrad = (proposals[k][nodes[2]] - proposals[i][nodes[0]])/2;

     if ((std::abs(Fx) <= kappa) && (std::abs(depGrad) <= alphaMax)) {
      sparseEnergies[cliqInd][energyInd] = -1*energyWt*std::abs(Fx);
      //std::cout<<sparseEnergy[energyInd]<<" ";
      ++sparseCnt;
      if (std::abs(Fx) <= pow(10,-6)) {
       ++zeroCnt;
      }
     }

     ++energyInd;
    }
   }
  }

  if (sparseCnt > maxSparseSiz) {
   maxSparseSiz = sparseCnt;
  }
  if (sparseCnt < minSparseSiz) {
   minSparseSiz = sparseCnt;
  }

  //std::cout<<"popHighStereoEnergies: clique index "<<cliqInd<<" max. sparse size "<<sparseCnt<<std::endl;
  //std::cout<<std::flush;

  ++cliqInd;
 }

 std::cout<<"popHighStereoEnergies: max. sparse size "<<maxSparseSiz<<" min. sparse size "<<minSparseSiz<<std::endl;

 return 0;
}

std::vector<double> birchTomasDisp(const int rPos, const int cPos, const int nRow, const int nCol, const int nLabel, const std::vector<double> &lImg, const std::vector<double> &rImg)
{
  std::vector<double> dataVec;

  int iL = 0;

  while ((iL < nLabel) && ((cPos + iL) < nCol)) {
   std::vector<double> rVal;

   int pixPos = rPos*nCol + cPos;

   rVal.push_back(rImg[pixPos]);

   int pixPosTwo;

   if (cPos < nCol - 1) {
    pixPosTwo = rPos*nCol + (cPos+1);

    rVal.push_back((rImg[pixPos] + rImg[pixPosTwo])/2);
   }
   else {
    rVal.push_back(rImg[pixPos]);
   }

   if (cPos > 0) {
    pixPosTwo = rPos*nCol + (cPos-1);

    rVal.push_back((rImg[pixPos] + rImg[pixPosTwo])/2);
   }
   else {
    rVal.push_back(rImg[pixPos]);
   }

   double Imax = *std::max_element(rVal.begin(), rVal.end());
   double Imin = *std::min_element(rVal.begin(), rVal.end());

   double lVal;

   if (cPos + iL < 0) {
    lVal = lImg[pixPos];
   }
   else {
    pixPosTwo = rPos*nCol + (cPos + iL);

    lVal = lImg[pixPosTwo];
   }

   std::vector<double> opSelect;

   opSelect.push_back(lVal - Imax);
   opSelect.push_back(Imin - lVal);
   opSelect.push_back(0);

   dataVec.push_back(-1*(*std::max_element(opSelect.begin(), opSelect.end())));

   ++iL;
  }

  int extraSz = nLabel - dataVec.size();

  if (extraSz != 0) {
   double dataBuff = dataVec.back();
   for (int i = 0; i != extraSz; ++i) {
    dataVec.push_back(dataBuff);
   }
  }

  return dataVec;
}
