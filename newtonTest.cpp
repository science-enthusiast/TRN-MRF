#include "dualSys.hpp"
#include "myUtils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <ctime>
#include <cmath>
#include "variousEnergies.hpp"

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/sparsemarray.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/pottsg.hxx>
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include <opengm/inference/external/ad3.hxx>

//std::vector<double> popUniformUEnergy(int);
//std::vector<double> popUniformCEnergy(int);
//std::vector<double> popUniformCEnergy(std::vector<int>, std::vector<short>);
//std::vector<double> popl1DispUEnergy(int, int, int, int, int, std::vector<double>, std::vector<double>);
//int popStereoSmoothEnergies(int, int, const std::vector<double> &, const std::vector<double> &, double, double, std::vector<std::vector<double> > &, double &, std::vector<std::map<int,double> >  &, std::vector<std::set<int> > &, std::vector<std::vector<int> > &);
void populateCliqLab(int, int, std::vector<std::vector<int> > &);
//std::vector<double> popDenoiseUEnergy(int, double);
//std::vector<double> popFoEUEnergy(int, double);
//std::vector<double> popSparseDenoiseCEnergy(int);
//int popSparseDenoiseCEnergy(int, int, double &, std::vector<std::set<int> > &, std::vector<std::map<int,double> > &);
//std::vector<std::pair<std::vector<short>,double> > popSparseDenoiseCEnergyPair(short);
//std::vector<double> popDenoiseCEnergy(int, int, std::vector<std::vector<int> >);
//std::vector<double> popTruncVarDenoiseCEnergy(int, int, std::vector<std::vector<int> >);
//std::vector<double> popFoECEnergy(int, std::vector<double>);
//std::vector<double> popSqCEnergy(int nLabel);

int populateNodeCliqLists(int, int, int, int, std::vector<std::vector<int> > &);

int main(int argc, char* argv[])
{
 std::string ipFile, ipStereoFile;
 int nNode;
 std::vector<short> nLabel;
 int totCliq, nCliq = 0; //totCliq = no. of nodes + cliques
 int sizCliq, nCliqLab;
 int numLabel, nRow, nCol, rCliq, cCliq, sepCliq, nRowCliq, nColCliq;
 std::vector<std::vector<int> > cliqNodes;

 std::vector<std::vector<double> > uEnergies;
 std::vector<double> sparseKappa;
 std::vector<std::set<int> > sparseIndices;
 std::vector<std::map<int,double> > sparseEnergies;
 double sparseKappaCom;
 std::set<int> sparseIndicesCom;
 std::map<int,double> sparseEnergiesCom;

 std::vector<double> curCEnergy;
 std::vector<std::vector<double> > cEnergyVec;

 std::string ipType, algoName, logLabel, mrfModel;
 bool sparseFlag;
 double sampN;
 bool oneCliqTable = false;

 dualSys* myDual;

 double tau = 1;
 int maxIter = 10000, annealIval = 1; //annealIval = -1: no annealing
 double stgCvxCoeff = 1;
 bool stgCvxFlag = false;
 bool stgCvxIp = false; //what is specified in the config file; accept only if fista or scd

 double unaryScale = 1;

 typedef opengm::Adder OperatorType;
 typedef opengm::DiscreteSpace<std::size_t, std::size_t> SpaceType;

 // Set functions for graphical model
 typedef opengm::meta::TypeListGenerator<
         opengm::ExplicitFunction<double, std::size_t, std::size_t>,
         opengm::PottsFunction<double, std::size_t, std::size_t>,
         opengm::PottsNFunction<double, std::size_t, std::size_t>,
         opengm::PottsGFunction<double, std::size_t, std::size_t>,
         opengm::TruncatedSquaredDifferenceFunction<double, std::size_t, std::size_t>,
         opengm::TruncatedAbsoluteDifferenceFunction<double, std::size_t, std::size_t>,
         opengm::SparseFunction<double, std::size_t, std::size_t>
         >::type FunctionTypeList;

 typedef opengm::GraphicalModel<
   double,
   OperatorType,
   FunctionTypeList,
   SpaceType
   > GmType;

 GmType gm;

 typedef opengm::external::AD3Inf<GmType,opengm::Minimizer> AD3SolverType;

// if (argc < 6) {
//  std::cout<<"USAGE: ./newtonTest <i/p file name> <no. of clique tables: 1 (one global) or 0 (individual)> <method N (Newton) F (Fista) L (LM)><input: uai y or in program n> <output flag>"<<std::endl;
//  return -1;
// }

 if (argc < 2) {
  std::cout<<"USAGE: ./newtonTest <name of configuration file>"<<std::endl;

  return -1;
 }

 std::ifstream fin(argv[1]);
 std::string line;
 std::istringstream sin;

 while (std::getline(fin, line)) {
  std::string varValue(line.substr(line.find("=")+1));
  //std::transform(varValue.begin(), varValue.end(), varValue.begin(), ::tolower);

  sin.str(varValue);

  if (line.find("Input name") != std::string::npos) {
   sin >> ipFile;
   std::cout<<"Input image: "<<ipFile<<std::endl;
  }
  else if (line.find("Stereo right image") != std::string::npos) {
   sin >> ipStereoFile;
   std::cout<<"Stereo right image: "<<ipStereoFile<<std::endl;
  }
  else if (line.find("No. of rows") != std::string::npos) {
   sin >> nRow;
   std::cout<<"No. of rows "<<nRow<<" ";
  }
  else if (line.find("No. of cols") != std::string::npos) {
   sin >> nCol;
   std::cout<<"No. of columns "<<nCol<<" ";
  }
  else if (line.find("No. of labels") != std::string::npos) {
   sin>>numLabel;
   std::cout<<"No. of labels "<<numLabel<<" ";
  }
  else if (line.find("Clique row size") != std::string::npos) {
   sin>>rCliq;
   std::cout<<"No. of rows in clique "<<rCliq<<" ";
  }
  else if (line.find("Clique col size") != std::string::npos) {
   sin>>cCliq;
   std::cout<<"No. of columns in clique "<<cCliq<<" ";
  }
  else if (line.find("Clique stride") != std::string::npos) {
   sin>>sepCliq;
  }
  else if (line.find("tau") != std::string::npos) {
   sin>>tau;
   std::cout<<"Initial tau value "<<tau<<std::endl;
  }
  else if (line.find("strong convexity coeff") != std::string::npos) {
   sin>>stgCvxCoeff;
   std::cout<<"Strong convexity coefficient "<<stgCvxCoeff<<std::endl;
  }
  else if (line.find("Strong convexity flag") != std::string::npos) {
   sin>>std::boolalpha>>stgCvxIp;
   std::cout<<"Strong convexity flag "<<stgCvxIp<<std::endl;
  }
  else if (line.find("Max. iterations") != std::string::npos) {
   sin>>maxIter;
   std::cout<<"Max. iterations "<<maxIter<<std::endl;
  }
  else if (line.find("Input type") != std::string::npos) {
   sin>>ipType;
   std::cout<<"Input type "<<ipType<<std::endl;
  }
  else if (line.find("Algorithm") != std::string::npos) {
   sin>>algoName;

   std::cout<<"Algorithm "<<algoName<<std::endl;
  }
  else if (line.find("Sparse") != std::string::npos) {
   sin>>std::boolalpha>>sparseFlag;
  }
  else if (line.find("Sample") != std::string::npos) {
   sin>>sampN;
  }
  else if (line.find("Anneal") != std::string::npos) {
   std::string annealFlag;

   sin>>annealFlag;

   if (annealFlag.compare("true") == 0) {
    annealIval = 1;
   }
   else if (annealFlag.compare("false") == 0) {
    annealIval = -1;
   }

   std::cout<<"Anneal flag "<<annealFlag<<std::endl;
  }
  else if (line.find("Log label") != std::string::npos) {
   sin>>logLabel;
  }
  else if (line.find("One clique table") != std::string::npos) {
   sin>>std::boolalpha>>oneCliqTable;

   std::cout<<"One clique Table flag: "<<std::boolalpha<<oneCliqTable<<std::endl;
  }
  else if (line.find("MRF model") != std::string::npos) {
   sin>>mrfModel;

   std::cout<<"MRF model: "<<mrfModel<<std::endl;
  }

  sin.clear();
 }

 algoName.assign(argv[2]);

 logLabel += algoName;

 std::cout<<"Log label: "<<logLabel<<std::endl;

 if ((algoName.compare("fista") != 0) && (stgCvxIp)) {
  std::cout<<"ERROR: strong convexity can be applied only with FISTA."<<std::endl;
  return -1;
 }
 else {
  stgCvxFlag = stgCvxIp;
 }

 if (ipType.compare("uai") == 0) { //INPUT: take input from uai file
  std::ifstream uaiFile(ipFile);

  if (uaiFile) {
   std::cout<<"Input file read success."<<std::endl;
  }
  else {
   std::cout<<"Input file read error."<<std::endl;
   return -1;
  }

  std::string curLine;
  std::istringstream sin;

  for (int i = 0; i != 2; ++i) {
   std::getline(uaiFile,curLine);
  }

  sin.str(curLine);
  sin>>nNode;

  nLabel.resize(nNode);

  sin.clear();
  std::getline(uaiFile,curLine);
  sin.str(curLine);

  for (int i = 0; i != nNode; ++i) {
   sin>>nLabel[i];
  }

  sin.clear();
  std::getline(uaiFile,curLine);
  sin.str(curLine);
  sin>>totCliq;

  myDual = new dualSys(nNode, nLabel, tau, stgCvxCoeff, maxIter, annealIval, stgCvxFlag);

  bool readNodeList = true;

  while (readNodeList) {
   std::getline(uaiFile,curLine);
   if (curLine.empty()) { //this takes care blank line at the end of this section
    readNodeList = false;
   }
   else {
    sin.clear();
    sin.str(curLine);
    //std::cout<<"current line "<<curLine<<std::endl;
    sin>>sizCliq;

    //std::cout<<"clique size "<<sizCliq<<std::endl;

    int nodeInd;
    std::vector<int> curNodes;

    while (sin>>nodeInd) {
     curNodes.push_back(nodeInd);
    }

    cliqNodes.push_back(curNodes);
   }
  }

  std::set<int> nodeSet; //set of nodes which has unary values explicitly specified

  nCliq = 0;

  if (!oneCliqTable) {
   for (std::vector<std::vector<int> >::iterator cliqInd = cliqNodes.begin(); cliqInd != cliqNodes.end(); ++cliqInd) {
    if ((*cliqInd).size() != 1) {
     ++nCliq;
    }
   }
  }

  cEnergyVec.resize(nCliq);

  nCliq = 0;

  bool cliqTablePopFlag = false;

  for (std::vector<std::vector<int> >::iterator cliqInd = cliqNodes.begin(); cliqInd != cliqNodes.end(); ++cliqInd) {
   std::getline(uaiFile,curLine);

   sin.clear();
   sin.str(curLine);
   sin>>nCliqLab;
   //std::cout<<"no. of labelings "<<nCliqLab<<" ";

   if ((*cliqInd).size() == 1) { //Clique could be one node. Unaries to be populated.
    std::getline(uaiFile,curLine);
    sin.clear();
    sin.str(curLine);

    double uVal;
    std::vector<double> uEnergy;

    while (sin>>uVal) {
     //std::cout<<uVal<<" ";

     double logVal = log(uVal);
     myUtils::checkLogError(logVal);

     uEnergy.push_back(logVal);
    }

    nodeSet.insert((*cliqInd)[0]);

    myDual->addNode((*cliqInd)[0],uEnergy);
   }
   else if ((*cliqInd).size() == 2) {//pair of nodes. Then clique energies are specified in multiple rows.
    if (oneCliqTable) {
     if (!cliqTablePopFlag) {
      curCEnergy.clear();

      for (int l = 0; l != nLabel[(*cliqInd)[0]]; ++l) {
       std::getline(uaiFile,curLine);
       sin.clear();
       sin.str(curLine);

       for (int u = 0; u != nLabel[(*cliqInd)[1]]; ++u) {
        double curC;
        sin>>curC;

        double logVal = log(curC);
        myUtils::checkLogError(logVal);

        curCEnergy.push_back(logVal);
        //curCEnergy[l*nLabel[(*cliqInd)[0]] + u] = logVal;
        //curCEnergy.push_back(logVal);
       }
      }

      cliqTablePopFlag = true;
     }
    }
    else {
     curCEnergy.clear();

     for (int l = 0; l != nLabel[(*cliqInd)[0]]; ++l) {
      std::getline(uaiFile,curLine);
      sin.clear();
      sin.str(curLine);

      for (int u = 0; u != nLabel[(*cliqInd)[1]]; ++u) {
       double curC;
       sin>>curC;

       double logVal = log(curC);
       myUtils::checkLogError(logVal);

       curCEnergy.push_back(logVal);
       //curCEnergy[l*nLabel[(*cliqInd)[0]] + u] = logVal;
       //curCEnergy.push_back(logVal);
      }
     }

     cEnergyVec[nCliq] = curCEnergy;
    }

    ++nCliq;
   }
   else {//more than two nodes. Clique energies in one big row
    if (oneCliqTable) {
     if (!cliqTablePopFlag) {
      curCEnergy.clear();

      std::getline(uaiFile,curLine);
      sin.clear();
      sin.str(curLine);

      for (int l = 0; l != nCliqLab; ++l) {
       double curC;
       sin>>curC;

       double logVal = log(curC);
       myUtils::checkLogError(logVal);

       curCEnergy.push_back(logVal);
      }

      cliqTablePopFlag = true;
     }
    } //if oneCliqTable
    else {
     curCEnergy.clear();

     std::getline(uaiFile,curLine);
     sin.clear();
     sin.str(curLine);

     for (int l = 0; l != nCliqLab; ++l) {
      double curC;
      sin>>curC;

      double logVal = log(curC);
      myUtils::checkLogError(logVal);

      curCEnergy.push_back(logVal);
     }

     cEnergyVec[nCliq] = curCEnergy;
    } //else oneCliqTable

    ++nCliq;
   }

   std::getline(uaiFile,curLine); //why is this here?
  }

  if (oneCliqTable) {
   nCliq = 0;

   for (std::vector<std::vector<int> >::iterator cliqInd = cliqNodes.begin(); cliqInd != cliqNodes.end(); ++cliqInd) {
    if ((*cliqInd).size() != 1) {//pair of nodes. Then clique energies are specified in multiple rows.
     myDual->addCliq(*cliqInd, &curCEnergy);

     ++nCliq;
    }
   }
  }
  else {
   nCliq = 0;

   for (std::vector<std::vector<int> >::iterator cliqInd = cliqNodes.begin(); cliqInd != cliqNodes.end(); ++cliqInd) {
    if ((*cliqInd).size() != 1) {//pair of nodes. Then clique energies are specified in multiple rows.
     myDual->addCliq(*cliqInd, &cEnergyVec[nCliq]);

     ++nCliq;
    }
   }
  }

  for (int n = 0; n != nNode; ++n) {
   if (nodeSet.find(n) == nodeSet.end()) {
    std::vector<double> uEnergy;
    for (int u = 0; u != nLabel[n]; ++u) {
     uEnergy.push_back(0); //log(1)
    }

    myDual->addNode(n,uEnergy);
   }
  }
 } //if ipFlag
 else if (ipType.compare("hdf5") == 0) {
  opengm::hdf5::load(gm,ipFile,"gm");

  if (algoName.compare("ad3") != 0) {
   nNode = gm.numberOfVariables();

   std::cout<<"Number of nodes "<<nNode<<std::endl;

   nLabel.resize(nNode);

   for (int n = 0; n != nNode; ++n) {
    nLabel[n] = gm.numberOfLabels(n);
   }

   myDual = new dualSys(nNode, nLabel, tau, stgCvxCoeff, maxIter, annealIval, stgCvxFlag);

   std::cout<<"Number of factors "<<gm.numberOfFactors()<<std::endl;

   if (sparseFlag) {
    sparseKappa.resize(gm.numberOfFactors());
    sparseEnergies.resize(gm.numberOfFactors());
    sparseIndices.resize(gm.numberOfFactors());
   }
   else {
    cEnergyVec.resize(gm.numberOfFactors());
   }

   //std::cout<<"newtonTest: unary and clique energies are "<<std::endl;

   for (std::size_t f = 0; f != gm.numberOfFactors(); ++f) {
    if (gm[f].numberOfVariables() == 1) {

     std::size_t l[1] = {0};
     std::vector<double> uEnergy;
     for (l[0] = 0; l[0] != gm[f].numberOfLabels(0); ++l[0]) {
      uEnergy.push_back(-1*unaryScale*gm[f](l));
      //std::cout<<uEnergy.back()<<" ";
     }

     myDual->addNode(gm[f].variableIndex(0),uEnergy);
    } //unary potentials
    else {
     if (sparseFlag) { //the clique energies are sparse. Each clique will have its own copy of clique energies.
      sparseKappa[f] = 10000;
      sparseEnergies[f].clear();
      sparseIndices[f].clear();
      curCEnergy.clear();

      double maxEnergy = -10000;
      double minEnergy = 10000;

      int sizCliq = gm[f].numberOfVariables();
      rCliq = 1;
      cCliq = sizCliq;

      std::vector<int> cliqVar(sizCliq);

      int nCliqLab = 1;

      for (int i = 0; i != sizCliq; ++i) {
       nCliqLab *= gm[f].numberOfLabels(i);
       cliqVar[i] = gm[f].variableIndex(i);
      }

      for (int iCliqLab = 0; iCliqLab != nCliqLab; ++iCliqLab) {
       std::size_t* lab = new std::size_t[sizCliq];

       double labPull = nCliqLab;

       for (int j = 0; j != sizCliq; ++j) {
        int curNumLab = gm[f].numberOfLabels(j);

        labPull /= curNumLab;

        int labPullTwo = ceil((iCliqLab+1)/labPull);

        if (labPullTwo % curNumLab == 0) {
         lab[j] = curNumLab - 1;
        }
        else {
         lab[j] = (labPullTwo % curNumLab) - 1;
        }
       }

       double curCEVal = -1*gm[f](lab); //performing primal maximization; dual minimization.

       curCEnergy.push_back(curCEVal);

       if (curCEVal > maxEnergy) {
        maxEnergy = curCEVal;
       }

       if (curCEVal < minEnergy) {
        minEnergy = curCEVal;
       }

       if (sparseKappa[f] > curCEVal) {
        sparseKappa[f] = curCEVal;
       }

       delete []lab;
      } //for iCliqLab

      for (int iCliqLab = 0; iCliqLab != nCliqLab; ++iCliqLab) {
       //std::cout<<curCEnergy[iCliqLab]<<" ";
       if (curCEnergy[iCliqLab] != sparseKappa[f]) {
        sparseEnergies[f][iCliqLab] = curCEnergy[iCliqLab];
        sparseIndices[f].insert(iCliqLab);
       }
      } //for iCliqLab

      //std::cout<<" "<<sparseIndices[f].size()<<" max "<<maxEnergy<<" min "<<minEnergy<<" sparse kappa "<<sparseKappa[f];

      std::cout<<f<<" clique variables are";
      for (std::vector<int>::const_iterator iVar = cliqVar.begin(); iVar != cliqVar.end(); ++iVar) {
       std::cout<<" "<<*iVar;
      }
      std::cout<<std::endl;

      myDual->addCliq(cliqVar, rCliq, cCliq, &sparseKappa[f], &sparseEnergies[f], &sparseIndices[f]);
     } //if sparseFlag
     else { //clique potentials are stored in a dense manner
      int sizCliq = gm[f].numberOfVariables();
      std::vector<int> cliqVar(sizCliq);

      int nCliqLab = 1;

      for (int i = 0; i != sizCliq; ++i) {
       nCliqLab *= gm[f].numberOfLabels(i);
       cliqVar[i] = gm[f].variableIndex(i);
      }

//      if (oneCliqTable) {
//       curCEnergy.clear();

      cEnergyVec[f].clear();

      for (int iCliqLab = 0; iCliqLab != nCliqLab; ++iCliqLab) {
       std::size_t* lab = new std::size_t[sizCliq];

       double labPull = nCliqLab;

       for (int j = 0; j != sizCliq; ++j) {
        int curNumLab = gm[f].numberOfLabels(j);

        labPull /= curNumLab;

        int labPullTwo = ceil((iCliqLab+1)/labPull);

        if (labPullTwo % curNumLab == 0) {
         lab[j] = curNumLab - 1;
        }
        else {
         lab[j] = (labPullTwo % curNumLab) - 1;
        }
       }

       cEnergyVec[f].push_back(-gm[f](lab));
      } //for iCliqLab

//       oneCliqTable = false;
//      }

      myDual->addCliq(cliqVar, &cEnergyVec[f]);
     }
    } //clique potentials

    //std::cout<<std::endl;
   }
  }
 }
 else { //INPUT: energies defined in code
  std::vector<double> pixVals;
  std::vector<double> pixValsStereo;

  sizCliq = rCliq*cCliq;
  nNode = nRow*nCol;
  nCliqLab = pow(numLabel,sizCliq);

  for (int i = 0 ; i != nNode; ++i) {
   nLabel.push_back(numLabel);
  }

  //"var": variance; "spden1": truncated variance, spare Denoise approach 1; "spden2": truncated higher-order smoothness, approach 2; "spstereo": sparse stereo

  if ((rCliq == 2) && (cCliq == 2)) {
   nRowCliq = nRow - rCliq + 1;
   nColCliq = nCol - cCliq + 1;
   nCliq = nRowCliq*nColCliq;

   std::cout<<"no. of cliques "<<nCliq<<std::endl;
   populateNodeCliqLists(rCliq, cCliq, nRow, nCol, cliqNodes);
   std::cout<<"cliqNodes size is "<<cliqNodes.size()<<std::endl;
  }
  else if ((mrfModel.compare("spden1") == 0) || (mrfModel.compare("spden2") == 0) || (mrfModel.compare("spden3") == 0) || (mrfModel.compare("sqdiff") == 0) || (mrfModel.compare("spstereo") == 0)) {
   nRowCliq = nRow - cCliq + 1;
   nColCliq = nCol - cCliq + 1;
   nCliq = nRowCliq*nCol + nColCliq*nRow;

   std::cout<<"no. of cliques "<<nCliq<<std::endl;
   populateNodeCliqLists(rCliq, cCliq, nRow, nCol, cliqNodes);
   std::cout<<"cliqNodes size is "<<cliqNodes.size()<<std::endl;
   populateNodeCliqLists(cCliq, rCliq, nRow, nCol, cliqNodes);
   std::cout<<"cliqNodes size is "<<cliqNodes.size()<<std::endl;
  }
  else if (mrfModel.compare("spgraphmatch") == 0) {
   std::string quPtsFile = "/home/hari/softwares/2d_graph_match/" + ipFile + "ptsqu.txt";

   std::ifstream ipPtsTuple(quPtsFile.c_str());

   std::string curPtsString;

   while (std::getline(ipPtsTuple,curPtsString)) {
    std::stringstream curPtsStream(curPtsString);

    std::vector<int> curPts(3);
    curPtsStream>>curPts[0];
    curPtsStream>>curPts[1];
    curPtsStream>>curPts[2];

    //std::cout<<curPts[0]<<" "<<curPts[1]<<" "<<curPts[2]<<std::endl;

    cliqNodes.push_back(curPts);
   }

   nCliq = cliqNodes.size();

   std::cout<<"no. of cliques "<<nCliq<<std::endl;
  }

  std::ifstream ipImgFile(ipFile);
  std::string imgRow;

  if ((mrfModel.compare("random") != 0) && (mrfModel.compare("sprandom") != 0) && (mrfModel.compare("spgraphmatch") != 0)) {
   if (ipImgFile) {
    while (ipImgFile>>imgRow) {
     std::stringstream imgRowStream(imgRow);

     double pixVal;

     while (imgRowStream>>pixVal) {
      pixVals.push_back(pixVal);
     }
    }

    ipImgFile.close();
   }
   else {
    std::cout<<"input file read error: "<<ipFile<<std::endl;
    return -1;
   }

   std::cout<<"pixel values loaded."<<std::endl;
  }

  //cliqNodes.resize(nCliq);

  myDual = new dualSys(nNode, nLabel, tau, stgCvxCoeff, maxIter, annealIval, stgCvxFlag);

  //adding Unaries
  if ((mrfModel.compare("var") == 0) || (mrfModel.compare("spden1") == 0) || (mrfModel.compare("spden2") == 0) || (mrfModel.compare("spden3") == 0) || (mrfModel.compare("sqdiff") == 0) || (mrfModel.compare("spvar") == 0) || (mrfModel.compare("spvarcom") == 0)) {
   for (int i = 0; i != nNode; ++i) {
    std::vector<double> uEnergy = popDenoiseUEnergy(nLabel[i], pixVals[i]);

    myDual->addNode(i,uEnergy);
   }
  }
  else if ((mrfModel.compare("random") == 0) || (mrfModel.compare("sprandom") == 0)) {
   for (int i = 0; i != nNode; ++i) {
    std::vector<double> uEnergy = popUniformUEnergy(nLabel[i]);

    myDual->addNode(i,uEnergy);
   }
  }
  else if (mrfModel.compare("spgraphmatch") == 0) {
   for (int i = 0; i != nNode; ++i) {
    std::vector<double> uEnergy = popStrongGraphMatchUEnergy(i,numLabel);
    //std::vector<double> uEnergy(numLabel,0);

    myDual->addNode(i,uEnergy);
   }
  }
  else {
   ipImgFile.open(ipStereoFile);

   if (ipImgFile) {
    while (ipImgFile>>imgRow) {
     std::stringstream imgRowStream(imgRow);

     double pixVal;

     while (imgRowStream>>pixVal) {
      pixValsStereo.push_back(pixVal);
     }
    }

    ipImgFile.close();
   }
   else {
    std::cout<<"stereo right image read error."<<std::endl;
    return -1;
   }

  //only pixel values of right stereo image loaded.
  //unaries will be added at a later stage for stereo.
  }

  std::cout<<"nodes populated."<<std::endl;

  std::set<std::vector<int> > checkCliqNodes;

  //var: variance prior
  //spden1: truncated variance prior
  if (mrfModel.compare("spden2") == 0) { //truncated higher-order smoothness prior
   //curCEnergyPair = popSparseDenoiseCEnergyPair(numLabel);
   curCEnergy = popSparseDenoiseCEnergy(numLabel);
  }
  else if (mrfModel.compare("spden3") == 0) {
   sparseIndices.resize(nCliq);
   sparseEnergies.resize(nCliq);
   sparseKappa.resize(nCliq);

   for (int iCliq = 0; iCliq != nCliq; ++iCliq) {
    popSparseDenoiseCEnergy(numLabel, sparseKappa[iCliq], sparseEnergies[iCliq]);
   }
  }
  else if (mrfModel.compare("spvar") == 0) {
   sparseIndices.resize(nCliq);
   sparseEnergies.resize(nCliq);
   sparseKappa.resize(nCliq);

   for (int iCliq = 0; iCliq != nCliq; ++iCliq) {
    popSparseVarCEnergy(sizCliq, nLabel, nCliqLab, sparseKappa[iCliq], sparseEnergies[iCliq]);
   }
  }
  else if (mrfModel.compare("spvarcom") == 0) {
   popSparseVarCEnergy(sizCliq, nLabel, nCliqLab, sparseKappaCom, sparseEnergiesCom);
  }
  else if (mrfModel.compare("sqdiff") == 0) {
   curCEnergy = popSqCEnergy(numLabel);
  }
  else if (mrfModel.compare("random") == 0) {
   curCEnergy = popUniformCEnergy(nCliqLab);
  }
  else if (mrfModel.compare("sprandom") == 0) {
   popSparseUniformCEnergy(nCliqLab, sparseKappaCom, sparseEnergiesCom);
  }
  else if (mrfModel.compare("spgraphmatch") == 0) {
   popSynth2DGraphMatchCEnergy(ipFile,nLabel,cliqNodes,sparseKappa, sparseEnergies);
  }
  else if (mrfModel.compare("spstereo") == 0) {
   std::cout<<"no. of cliques is "<<nCliq<<std::endl;
   for (int i = 0; i != nCliq; ++i) {
    checkCliqNodes.insert(cliqNodes[i]);
//    std::cout<<"clique "<<i<<" has ";
//    for (std::vector<int>::iterator nodeIter = cliqNodes[i].begin(); nodeIter != cliqNodes[i].end(); ++nodeIter) {
//     std::cout<<*nodeIter<<" ";
//    }
//    std::cout<<std::endl;
   }

   cliqNodes.clear();

   double gtScale = 0.125, downScale = 0.25;

   popStereoSmoothEnergies(nRow, nCol, pixVals, pixValsStereo, gtScale, downScale, uEnergies, sparseKappaCom, sparseEnergies, cliqNodes);

   for (int y = 0; y != nRow; ++y) {
    for (int x = 0; x != nCol; ++x) {
     int pixInd = y*nCol + x;

     myDual->addNode(pixInd,uEnergies[pixInd]);
    }
   }
  } //else if spstereo

  if (sparseFlag) {
   if  (oneCliqTable) {
    sparseIndicesCom.clear();

    for (std::map<int,double>::const_iterator iEnergy = sparseEnergiesCom.begin(); iEnergy != sparseEnergiesCom.end(); ++iEnergy) {
     sparseIndicesCom.insert(iEnergy->first);
    }
   }
   else {
    sparseIndices.clear();

    for (std::vector<std::map<int,double> >::const_iterator iSpEnergy = sparseEnergies.begin(); iSpEnergy != sparseEnergies.end(); ++iSpEnergy) {
     std::map<int,double> curSpEnergy = *iSpEnergy;
     std::set<int> curIndices;

     for (std::map<int,double>::const_iterator iCur = curSpEnergy.begin(); iCur != curSpEnergy.end(); ++iCur) {
      curIndices.insert(iCur->first);
     }

     sparseIndices.push_back(curIndices);
    }
   }
  }

  std::cout<<std::flush;

  std::cout<<"sparse clique energies:";

  for (int i = 0; i != nCliq; ++i) {
   //std::cout<<" "<<i;
//   for (std::vector<int>::iterator nodeIter = cliqNodes[i].begin(); nodeIter != cliqNodes[i].end(); ++nodeIter) {
//    std::cout<<*nodeIter<<" ";
//   }
//   std::cout<<std::endl;
   if ((mrfModel.compare("var") == 0) || (mrfModel.compare("spden1") == 0) || (mrfModel.compare("spden2") == 0) || (mrfModel.compare("sqdiff") == 0) || (mrfModel.compare("random") == 0) ) {
    myDual->addCliq(cliqNodes[i],&curCEnergy);
   }
   else if ((mrfModel.compare("spstereo") == 0) || (mrfModel.compare("spden3") == 0) || (mrfModel.compare("spvar") == 0) || (mrfModel.compare("spgraphmatch") == 0)) {
    myDual->addCliq(cliqNodes[i], rCliq, cCliq, &sparseKappa[i], &sparseEnergies[i], &sparseIndices[i]);
    std::cout<<" "<<sparseIndices[i].size();
   }
   else if ((mrfModel.compare("spvarcom") == 0) || (mrfModel.compare("sprandom") == 0)) {
    myDual->addCliq(cliqNodes[i], rCliq, cCliq, &sparseKappaCom, &sparseEnergiesCom, &sparseIndicesCom);
   }
  }

  std::cout<<std::endl;

  std::cout<<"cliques populated."<<std::endl;
 }

 double tTotal = myUtils::getTime();

 std::vector<size_t> primalMax;

 if (algoName.compare("ad3") != 0) {
  std::cout<<"no. of label sets (one per node) "<<nLabel.size()<<std::endl;
  std::cout<<"no. of nodes (should be same as no. of label sets) "<<nNode<<std::endl;
  std::cout<<"no. of cliques "<<nCliq<<std::endl;

  myDual->prepareDualSys();

  if (algoName.compare("trn") == 0) {
   myDual->solveNewton();
  }
  else if (algoName.compare("fista") == 0) {
   myDual->solveFista();
  }
  else if (algoName.compare("scd") == 0) {
   myDual->solveSCD();
  }
  else {
   std::cout<<"Enter algorithm option correctly."<<std::endl;
   return -1;
  }

  primalMax = myDual->getPrimalMax();

  std::cout<<"Time taken to reach best integral primal "<<myDual->getTimeToBestPrimal()<<std::endl;
 }
 else if (algoName.compare("ad3") == 0) {
  delete myDual;

  nNode = gm.numberOfVariables();

  AD3SolverType::Parameter paraAD3;

  paraAD3.solverType_ = AD3SolverType::AD3_LP;

  paraAD3.steps_ = maxIter;

  AD3SolverType ad3(gm,paraAD3);

  AD3SolverType::VerboseVisitorType visitor;

  ad3.infer(visitor);

  ad3.arg(primalMax);

  std::cout<<"output labeling has energy: "<<gm.evaluate(primalMax)<<std::endl;
 }
 else {
  std::cout<<"Enter algorithm option correctly. trn Newton, fista FISTA, lm Levenberg-Marquardt, qn Quasi Newton"<<std::endl;
  return -1;
 }

 std::cout<<"Solved Newton System"<<std::endl;

 std::cout<<"Exiting! Total time "<<(myUtils::getTime() - tTotal)<<std::endl;

 std::size_t slashPos = ipFile.find_last_of('/');
 std::size_t dotPos = ipFile.find_last_of('.');

 std::size_t strLen = dotPos - slashPos - 1; //exclude the dot

 std::string opName = "test" + ipFile.substr(slashPos+1,strLen) + logLabel + ".txt";

 std::ofstream opImg(opName.c_str());

 std::cout<<"output file "<<opName<<" open status "<<opImg.is_open()<<std::endl;

 for (int i = 0; i != nNode - 1; ++i) {
  opImg<<primalMax[i]<<" ";
 }
 opImg<<primalMax[nNode - 1];
 opImg.close();

#if 0
 std::ifstream propFilesList("proposalFiles.txt");

 std::vector<std::vector<double> > proposals;

 std::string propFile;

 while (propFilesList>>propFile) {
  std::cout<<propFile<<std::endl;

  std::ifstream propStream(propFile);

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

 int numProposals = proposals.size();

 std::cout<<"number of proposals is "<<numProposals<<std::endl;

 std::string opStereoName = "opStereo" + ipFile.substr(slashPos+1,strLen) + "txt";

 std::ofstream opStereoImg(opStereoName.c_str());

 std::cout<<"output file "<<opStereoName<<" open status "<<opStereoImg.is_open()<<std::endl;

 for (int i = 0; i != nNode - 1; ++i) {
  opStereoImg<<proposals[primalMax[i]][i]<<" ";
 }
 opStereoImg<<proposals[primalMax[nNode - 1]][nNode-1];
 opStereoImg.close();
#endif

 return 0;
}


void populateCliqLab(int lev, int nLabel, std::vector<std::vector<int> > &cliqLab) {
 if (lev == 1) {
  std::cout<<"reached base case"<<std::endl;
  for (int i = 0; i != nLabel; ++i) {
   std::vector<int> t;
   t.push_back(i);
   cliqLab.push_back(t);
  }
 }
 else {
  populateCliqLab(lev-1, nLabel, cliqLab);

  std::vector<std::vector<int> > expandVec;

  for (std::vector<std::vector<int> >::iterator i = cliqLab.begin(); i != cliqLab.end(); ++i) {
   std::vector<int> curVec = *i;

   for (int i = 0; i != nLabel; ++i) {
    curVec.push_back(i);
    expandVec.push_back(curVec);
    curVec.pop_back();
   }
  }

  cliqLab = expandVec;
 }
}

int populateNodeCliqLists(int rCliq, int cCliq, int nRow, int nCol, std::vector<std::vector<int> > &cliqNodes) {

 if (((rCliq == 2) && (cCliq == 2)) && ((nRow % 2 != 0) || (nCol % 2 != 0))) {
  std::cout<<"populateNodeCliqLists: for clique of shape 2X2, make both row and column size of the image even."<<std::endl;

  return -1;
 }

 int nRowCliq;
 int nColCliq;
 int firstNodeOff;

 if ((rCliq == 2)&&(cCliq == 2)) {
  nRowCliq = nRow/2;
  nColCliq = nCol/2;
  firstNodeOff = 2;
 }
 else {
  nRowCliq = nRow - rCliq + 1;
  nColCliq = nCol - cCliq + 1;
  firstNodeOff = 1;
 }

 int nCliq = nRowCliq*nColCliq;

 int firstNode = 0;
 int rowCliqCnt = 0;
 int colCliqCnt = 1;

 for (int iCliq = 0; iCliq != nCliq; ++iCliq) {
  std::vector<int> curNodesVec;
  int rowNode = firstNode;

  for (int i = 0; i != rCliq; ++i) {
   for (int j = 0; j != cCliq; ++j) {
    int curNode = rowNode + j;
    curNodesVec.push_back(curNode);
   }

   rowNode += nCol;
  }

  cliqNodes.push_back(curNodesVec);

  if (colCliqCnt != nColCliq) {
   firstNode += firstNodeOff;
   ++colCliqCnt;
  }
  else {
   ++rowCliqCnt;
   firstNode = rowCliqCnt*nCol;
   colCliqCnt = 1;
  }
 }

 return 0;
}
