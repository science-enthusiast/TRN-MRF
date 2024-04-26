#include "myUtils.hpp"
#include <algorithm>
#include <vector>
namespace myUtils
{


double getTime()
{
 struct timeval fetchTime;
 double secT, muSecT;

 gettimeofday(&fetchTime, NULL);

 secT = static_cast<double>(fetchTime.tv_sec);
 muSecT = static_cast<double>(fetchTime.tv_usec);
 return (secT*1000 + muSecT/1000 + 0.5)/1000;
}

double minNonZeroAbsVal(const Eigen::VectorXd & ipVec, int vecSize)
{
 Eigen::VectorXd absVec = ipVec.array().abs();

 std::vector<double> stlVec(absVec.data(), absVec.data() + vecSize);

 double minVal = *std::max_element(stlVec.begin(), stlVec.end()); //initialize with maximum absolute value

 for (int i = 0; i != vecSize; ++i) {
  if ((stlVec[i] != 0) && (stlVec[i] < minVal)) {
   minVal = stlVec[i];
  }
 }

 return minVal;
}
 
}
