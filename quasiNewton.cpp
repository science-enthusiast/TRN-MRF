#include "quasiNewton.hpp"
#include <iostream>

quasiNewton::quasiNewton(int memSiz):offRing_(-2), memSiz_(memSiz)
{
 sArr_.resize(memSiz_);
 yArr_.resize(memSiz_);
 rho_.resize(memSiz_);
}

Eigen::VectorXd quasiNewton::solve(const Eigen::VectorXd &gradient, const Eigen::VectorXd &dualVar, const double &trRho, const int &cntIter)
{
 std::cout<<"Quasi newton solve entered. Iteration number is "<<cntIter<<" trust region parameter is "<<trRho<<std::endl;

 Eigen::VectorXd q = gradient;
 Eigen::VectorXd r;

 if (cntIter > memSiz_+1) {
  std::vector<double> alpha(memSiz_);
  double beta;

  for (int i = 0; i != memSiz_; ++i) {
   //std::cout<<"ring offset "<<offRing_<<std::endl;

   int iRing = offRing_ - i;

   if (iRing < 0) {
    iRing += memSiz_;
   }

   alpha[memSiz_-1-i] = rho_[iRing]*sArr_[iRing].dot(gradient);

   if (isnan(alpha[memSiz_-1-i])) {
    std::cout<<"alpha IS NAN!"<<std::endl;
   }

   q = q - alpha[memSiz_-1-i]*yArr_[iRing];

   for (int i = 0; i != q.size(); ++i) {
    if (isnan(q[i])) {
     std::cout<<"ELEMENT OF q IS NAN!"<<std::endl;

     std::cout<<"alpha is "<<alpha[memSiz_-1-i]<<std::endl;

     std::cout<<"y array elements are:"<<std::endl;
     for (int i = 0; i != yArr_[iRing].size(); ++i) {
      std::cout<<yArr_[iRing][i]<<std::endl;
     }
    }
   }
  }

  for (int i = 0; i != q.size(); ++i) {
   if (isnan(q[i])) {
    std::cout<<"ELEMENT OF q IS NAN!"<<std::endl;
   }
  }

  r = precond(q);

  for (int i = 0; i != r.size(); ++i) {
   if (isnan(r[i])) {
    std::cout<<"ELEMENT OF r IS NAN!"<<std::endl;
   }
  }
  for (int i = 0; i != memSiz_; ++i) {
   int iRing = offRing_ + i;

   if (iRing >= memSiz_) {
    iRing -= memSiz_;
   }

   beta = rho_[iRing]*yArr_[iRing].dot(gradient);

   r = r + (alpha[i] - beta)*sArr_[iRing];
  }

  for (int i = 0; i != r.size(); ++i) {
   if (isnan(r[i])) {
    std::cout<<"ELEMENT OF r IS NAN!"<<std::endl;
   }
  }

 }
 else {
  r = -1*gradient;
 }

 offRing_ += 1;

 if (offRing_ > memSiz_-1) {
  offRing_ = 0;
 }

 if (cntIter == 1) {
  offRing_ = -1;
  oldDualVar_ = dualVar;
  oldGrad_ = gradient;
 }
 else {
  std::cout<<"iteration count "<<cntIter<<" ring offset "<<offRing_<<std::endl;

  sArr_[offRing_] = dualVar - oldDualVar_;
  yArr_[offRing_] = gradient - oldGrad_ + trRho*sArr_[offRing_];
  rho_[offRing_] = 1/yArr_[offRing_].dot(sArr_[offRing_]);

  for (int i = 0; i != sArr_[offRing_].size(); ++i) {
   if (isnan(sArr_[offRing_][i])) {
    std::cout<<"ELEMENT OF sArr IS NAN!"<<std::endl;
   }
  }

  for (int i = 0; i != yArr_[offRing_].size(); ++i) {
   if (isnan(yArr_[offRing_][i])) {
    std::cout<<"ELEMENT OF yArr IS NAN!"<<std::endl;
   }
  }

  if (isinf(rho_[offRing_])) {
   std::cout<<"rho_ IS INF!"<<std::endl;

   std::cout<<"sArr_ ";
   for (int i = 0; i != sArr_[offRing_].size(); ++i) {
    std::cout<<sArr_[offRing_][i]<<" ";
   }
   std::cout<<std::endl;

   std::cout<<"yArr_ ";
   for (int i = 0; i != yArr_[offRing_].size(); ++i) {
    std::cout<<yArr_[offRing_][i]<<" ";
   }
  }
  std::cout<<std::endl;

  scaleParam_ = yArr_[offRing_].dot(sArr_[offRing_])/yArr_[offRing_].dot(yArr_[offRing_]);

  oldDualVar_ = dualVar;
  oldGrad_ = gradient;
 }

 return r;
}

Eigen::VectorXd quasiNewton::precond(Eigen::VectorXd q)
{
 Eigen::VectorXd r;

 if (isinf(scaleParam_)) {
  std::cout<<"Scale parameter is infinity."<<std::endl;
 }

 r = scaleParam_*q;

 return r;
}
