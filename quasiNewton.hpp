#include <vector>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

class quasiNewton
{
 std::vector<Eigen::VectorXd > sArr_;
 std::vector<Eigen::VectorXd > yArr_;
 std::vector<double> rho_;
 Eigen::VectorXd oldGrad_;
 Eigen::VectorXd oldDualVar_;
 int offRing_,memSiz_;
 double trRho_, scaleParam_;

 Eigen::SparseMatrix<double> precondMat;

 Eigen::VectorXd precond(Eigen::VectorXd);

 public:
  quasiNewton(int);
  Eigen::VectorXd solve(const Eigen::VectorXd &, const Eigen::VectorXd &, const double &, const int &);
};
