#include "dualSys.hpp"
#include "myUtils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <ctime>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/pottsg.hxx>
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"

int main(int argc, char* argv[])
{
 typedef opengm::Adder OperatorType;
 typedef opengm::DiscreteSpace<std::size_t, std::size_t> SpaceType;

 // Set functions for graphical model
 typedef opengm::meta::TypeListGenerator<
    opengm::ExplicitFunction<double, std::size_t, std::size_t>,
    opengm::PottsFunction<double, std::size_t, std::size_t>,
    opengm::PottsNFunction<double, std::size_t, std::size_t>,
    opengm::PottsGFunction<double, std::size_t, std::size_t>,
    opengm::TruncatedSquaredDifferenceFunction<double, std::size_t, std::size_t>,
    opengm::TruncatedAbsoluteDifferenceFunction<double, std::size_t, std::size_t>
 >::type FunctionTypeList;

 typedef opengm::GraphicalModel<
    double,
    OperatorType,
    FunctionTypeList,
    SpaceType
 > GmType;

 GmType gm; 

 int nNode;
 std::vector<short> nLabel;
 int totCliq, nCliq = 0; //totCliq = no. of nodes + cliques
 int sizCliq, nCliqLab;
 double tau = 1;
 int maxIter = 10000000, annealIval = 1; //annealIval = -1: no annealing

 std::string ipFile(argv[1]);

 std::ofstream opFile("debug_opengmDs.txt");

#if 0
 std::ifstream uaiFile(ipFile);
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

 dualSys* myDual = new dualSys(nNode, nLabel, tau, maxIter, annealIval);

 std::vector<std::vector<int> > cliqNodes;

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

 bool oneCliqTable = false;

 std::vector<double> curCEnergy;

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

   //std::cout<<std::endl;

   nodeSet.insert((*cliqInd)[0]);

   myDual->addNode((*cliqInd)[0],uEnergy); 
  }
  else if ((*cliqInd).size() == 2) {//pair of nodes. Then clique energies are specified in multiple rows.
   ++nCliq;

   if (!oneCliqTable) {
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
   }

   myDual->addCliq(*cliqInd, &curCEnergy);

   if (*argv[2] - '0' == 1) {
    oneCliqTable = true;
   }
  }
  else {//more than two nodes. Clique energies in one big row
   ++nCliq;

   if (!oneCliqTable) {
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
   }

   myDual->addCliq(*cliqInd, &curCEnergy);

   if (*argv[2] - '0' == 1) {
    oneCliqTable = true;
   }
  }

  //std::cout<<std::endl;

  std::getline(uaiFile,curLine);
 }

#else
 opengm::hdf5::load(gm,ipFile,"gm");
 
 nNode = gm.numberOfVariables();

 nLabel.resize(nNode);

 for (int n = 0; n != nNode; ++n) {
  nLabel[n] = gm.numberOfLabels(n);
 }

 dualSys* myDual = new dualSys(nNode, nLabel, tau, maxIter, annealIval);

 std::set<int> nodeSet; //set of nodes which has unary values explicitly specified

 std::cout<<"Number of factors "<<gm.numberOfFactors()<<std::endl;

 for (std::size_t f = 0; f != gm.numberOfFactors(); ++f) {
  if (gm[f].numberOfVariables() == 1) {
   std::size_t l[1] = {0};
   std::vector<double> uEnergy;
   //opFile<<gm[f].numberOfLabels(0)<<std::endl;
   for (l[0] = 0; l[0] != gm[f].numberOfLabels(0); ++l[0]) {
    uEnergy.push_back(-gm[f](l));
    //opFile<<exp(-gm[f](l))<<" ";
   }
   opFile<<"1 "<<gm[f].variableIndex(0)<<" ";
   opFile<<std::endl;

   myDual->addNode(gm[f].variableIndex(0),uEnergy);

   nodeSet.insert(gm[f].variableIndex(0));
  }
  else {
   int sizCliq = gm[f].numberOfVariables();
   std::vector<int> cliqVar(sizCliq);

   int totNumLab = 1;

   opFile<<sizCliq<<" ";

   for (int i = 0; i != sizCliq; ++i) {
    totNumLab *= gm[f].numberOfLabels(i);
    cliqVar[i] = gm[f].variableIndex(i);
    opFile<<gm[f].variableIndex(i)<<" ";
   }

   //opFile<<totNumLab<<std::endl;

   std::vector<double> curCEnergy;

   for (int i = 0; i != totNumLab; ++i) {
    std::size_t* lab = new std::size_t[sizCliq];

    double labPull = totNumLab;

    for (int j = 0; j != sizCliq; ++j) {
     int curNumLab = gm[f].numberOfLabels(j);

     labPull /= curNumLab;

     int labPullTwo = ceil((i+1)/labPull);

     if (labPullTwo % curNumLab == 0) {
      lab[j] = curNumLab - 1;
     }
     else {
      lab[j] = (labPullTwo % curNumLab) - 1;
     }
    }

    curCEnergy.push_back(-gm[f](lab));
    //opFile<<exp(-gm[f](lab))<<" ";
   }
   
   opFile<<std::endl;

   myDual->addCliq(cliqVar, &curCEnergy);
  }
 }
#endif

 opFile.close();

 for (int n = 0; n != nNode; ++n) {
  if (nodeSet.find(n) == nodeSet.end()) {
   std::vector<double> uEnergy;
   for (int u = 0; u != nLabel[n]; ++u) {
    uEnergy.push_back(0); //log(1)
   }
 
   myDual->addNode(n,uEnergy);
  }
 }

 std::cout<<"no. of label sets (one per node) "<<nLabel.size()<<std::endl;
 std::cout<<"no. of nodes (should be same as no. of label sets) "<<nNode<<std::endl;
 std::cout<<"no. of cliques "<<nCliq<<std::endl;

 myDual->prepareDualSys();

 double tTotal = myUtils::getTime();

 if (*argv[3] == 'N') {
  myDual->solveNewton();
 }
 else if (*argv[3] == 'F') {
  myDual->solveFista();
 } else if (*argv[3] == 'L') {
  myDual->solveTrueLM();
 }
 else {
  std::cout<<"Enter algorithm option correctly. N Newton, F FISTA, L Levenberg-Marquardt"<<std::endl;
  return -1;
 }

 std::cout<<"Solved optimization problem"<<std::endl;

 std::cout<<"Exiting! Total time "<<(myUtils::getTime() - tTotal)<<std::endl;

 std::vector<int> primalMax = myDual->getPrimalMax();

 std::size_t slashPos = ipFile.find_last_of('/');
 std::size_t dotPos = ipFile.find_last_of('.');

 std::size_t strLen = dotPos - slashPos;

 std::string opName = "test" + ipFile.substr(slashPos+1,strLen) + "txt";

 std::ofstream opImg(opName.c_str());

 std::cout<<"output file "<<opName<<" open status "<<opImg.is_open()<<std::endl;

 for (int i = 0; i != nNode - 1; ++i) {
  opImg<<primalMax[i]<<" ";
 }
 opImg<<primalMax[nNode - 1];
 opImg.close();

 return 0;
}
