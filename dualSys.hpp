#ifndef DUALSYS_HPP
#define DUALSYS_HPP

#include <vector>
#include <set>
#include <iostream>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include "subProblem.hpp"

extern "C"{
void precnd_(int *n, int *m, int* iop, int* iprob, int* jcg, double* s, double* y, double* r, double* z, double* w, int* lw, int* iw, int* liw, bool* build, int* info, int* mssg);
}

class dualSys {
 public:
  friend int matVecMult(const dualSys &, const std::vector<double> &, std::vector<double> &);
  dualSys(int, std::vector<short>, double, int, int);

  int addNode(int, std::vector<double>);
  int addCliq(const std::vector<int>&, std::vector<double> *);
  int addCliq(const std::vector<int> &, std::vector<std::pair<std::vector<short>,double> > *);
  int addCliq(const std::vector<int> &, double *, std::map<int,double> *, std::set<int> *);
  int addCliq(const std::vector<int> &, int, int, double *, std::map<int,double> *, std::set<int> *);
  int addCliq(const std::vector<int> &, const std::vector<double> &);
  int prepareDualSys();
  int solveNewton();
  int solveQuasiNewton();
  int solveFistaOri();
  int solveFista();
  int solveTrueLM();
  int solveSCD();

  std::vector<size_t> getPrimalMax() {return bestPrimalMax_;}
  double getTimeToBestPrimal() {return timeToBestPrimal_;}
  double compEnergy(std::vector<double>);
  double compEnergySparse(std::vector<double>);

  std::size_t nDualVar_;
  Eigen::VectorXi reserveHess_;

 private:
  std::vector<double> uEnergy_;
  std::vector<std::vector<double> > cEnergy2DVec_;
  std::vector<double> cEnergy_;
  std::vector<int> unaryOffset_;
  std::vector<int> cliqOffset_;

  int nNode_;
  int numLabTot_;
  std::vector<short> nLabel_;
  int nLabCom_;
  int nSubDualSizCom_;
  int nCliq_;
  int maxCliqSiz_;
  int nLabelMax_;
  std::size_t maxNumNeigh_;

  std::vector<int> cliqSizes_;

  std::vector<double> dualVar_;
  std::vector<double> momentum_;
  std::vector<double> gradient_;
  std::vector<double> newtonStep_;

  std::vector<subProblem> subProb_;

  std::vector<std::vector<int> > cliqPerNode_;

  std::vector<double> primalFrac_;
  std::vector<double> primalConsist_;
  std::vector<size_t> primalMax_;
  std::vector<size_t> bestPrimalMax_;
  double bestNonSmoothPrimalEnergy_;
  double timeToBestPrimal_;
  double timeStart_;

  double tau_, tauStep_;
  bool solnQual_;
  double curEnergy_;
  double finalEnergy_;

  bool sparseFlag_;
  bool pdInitFlag_;
  double pdGap_;
  int smallGapIterCnt_;

  int popGradHessEnergy(int);
  int popGradHessEnergyPerf(int);
  int popGradEnergyFista();
  int popSparseGradEnergyFista();
  int popGradEnergy();
  int popGradEnergy(const std::vector<double> &, std::vector<double> &, double &, double &, double &);
  int updateMSD(int);
  int updateStar(int);

  Eigen::SparseMatrix<double> eigenHess_;
  double *cHess_;

  Eigen::SparseMatrix<double> cliqBlkHess_;
  std::vector<Eigen::MatrixXd> nodeBlkHess_;
  std::vector<double*> nodeBlk_;
  Eigen::VectorXi reserveCliqBlk_;
  std::vector<Eigen::VectorXi> reserveNodeBlk_;
  std::vector<std::vector<int> > sparseNodeOffset_;

  Eigen::VectorXd eigenGrad_;

  int performLineSearch();
  int performCGBacktrack(std::vector<Eigen::VectorXd>);
  int performLineSearch(std::vector<double> &, double &);

  int distributeDualVars();
  int distributeMomentum();

  void recoverFracPrimal();
  void recoverFeasPrimal();
  void recoverNodeFracPrimal();
  void recoverMaxPrimal(std::vector<double>);
  void assignPrimalVars(std::string);
  int setFracAsFeas();

  double compNonSmoothDualEnergy();
  double compIntPrimalEnergy();
  double compSmoothPrimalEnergy();
  double compNonSmoothPrimalEnergy();

  //members for merged clique energies
  int prepareMergedCliques();
  double compIntPrimalEnergyMerge();
  std::vector<std::vector<double> > cEnergy2DVecMerge_;

  int solveCG(Eigen::VectorXd&, std::vector<Eigen::VectorXd>&, int);

  Eigen::VectorXd getHessVecProd(const Eigen::VectorXd &);
  Eigen::VectorXd getOptimHessVecProd(const Eigen::VectorXd &);
  Eigen::VectorXd getDistHessVecProd(const Eigen::VectorXd &);
  Eigen::VectorXd getExplicitHessVecProd(const Eigen::VectorXd &);

  int precondFlag_; //indicate which preconditioner is used

  //Jacobi preconditioner
  Eigen::VectorXd diagPrecond(Eigen::VectorXd);

  //block Jacobi preconditioner
  std::vector<Eigen::MatrixXd> blkJacob_;
  std::vector<Eigen::MatrixXd> blkDiag_;
  Eigen::VectorXd blkJacobPrecond(Eigen::VectorXd);

  //inc. Cholesky preconditioner
  int nNonZeroLower_; //std::size_t is preferable but incomplete cholesky library requires int
  std::vector<double> nzHessLower_;
  double* hessDiag_;
  int* hessColPtr_;
  std::vector<int> hessRowInd_;
  double* inCholLow_;
  double* inCholDiag_;
  int* inCholColPtr_;
  int* inCholRowInd_;
  int compInCholPrecond(int);
  Eigen::VectorXd applyInCholPrecond(Eigen::VectorXd);

  //Quasi-Newton preconditioner
  double* sVec_;
  double* yVec_;
  double* resVec_;
  double* prVec_;
  int numVarQN_, numPairQN_, iopQN_;
  double* workVecQN_;
  int* iWorkVecQN_;
  int lwQN_, liwQN_;
  bool buildQN_;

  double dampLambda_;
  static constexpr double gradTol_ = 0.001;
  static constexpr double dampTol_ = 0.001;
  int maxIter_;
  static constexpr double lsTol_ = 1e-6;//0.000001;
  static constexpr double lsC_ = 1e-4; //0.001; New values
  static constexpr double lsRho_ = 0.8; //0.9;
  static constexpr double dampLambdaInit_ = 0.1;
  static constexpr double tauScale_ = 2.0;
  static constexpr double gradDampFactor_ = 3;
  static constexpr double linSysScale_ = 1;
  static constexpr int cntExitCond_ = 20;
  static constexpr double tauMax_ = 8192;
  static constexpr double mcPrec_ = 1e-10; //std::numeric_limits<float>::denorm_min(); denorm_min is too small
  static constexpr double minNumAllow_ = 1e-14;
  static constexpr bool fullHessStoreFlag_ = false;

  double lsAlpha_;
  int annealIval_;
};

#endif //DUALSYSGEN_HPP
