#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

int main(int argc, char* argv[]) {

 std::ifstream fileList(argv[1]);

 std::string curName;

 while (std::getline(fileList,curName)) {
  std::ifstream ipFile(curName.c_str());

  //std::string serialNo = curName.substr(curName.find("ProtPred")+8,curName.find_last_of('_') - curName.find("ProtPred") - 8);
  std::string serialNo = curName.substr(0,curName.find_last_of('.'));

  std::string opName = "timeTaken_" + serialNo + ".txt";

  std::ofstream opFile(opName.c_str());
  std::stringstream sin;
  std::string curLine;

#if 0
  std::string findStringOne("solveNewton: gradient l-infinity norm: ");
  std::string findStringTwo("Energy: ");
  int numOff = 7;
#endif
#if 1
  std::string findStringOne("NOTE: ITERATION ");
  std::string findStringTwo("took");
  int numOff = findStringTwo.size();
#endif
#if 0
  std::string findStringOne("ITERATION ");
  std::string findStringTwo("took");
  int numOff = 5;
#endif
#if 0
  std::string findStringOne("Fractional primal energy: ");
  std::string findStringTwo("Integral primal energy:");
  int numOff = 24;
#endif
#if 0
  std::string findStringOne("Fractional primal energy: ");
  std::string findStringTwo("Non-smooth dual energy:");
  int numOff = 24;
#endif
#if 0
  std::string findStringOne("Fractional primal energy: ");
  std::string findStringTwo("Non-smooth dual energy:");
  int numOff = findStringTwo.size();
#endif

  while (std::getline(ipFile,curLine)) {
   if (curLine.find(findStringOne) != std::string::npos) {
    sin<<std::fixed;
    sin.precision(6);
    sin.str(curLine.substr(curLine.find(findStringTwo) + numOff));
    double val;
    sin>>val;

    opFile<<std::fixed;
    opFile<<std::setprecision(6);
    opFile<<val<<std::endl;
    sin.clear();
   }
  }

  ipFile.close();
  opFile.close();
 }

 return 0;
}
