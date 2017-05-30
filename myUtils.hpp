#ifndef MYUTILS_HPP
#define MYUTILS_HPP
#include <cmath>
#include <iostream>
#include <cerrno>
#include <sys/time.h>
#include <Eigen/Dense>

namespace myUtils
{

//int checkLogError(double&);
//int checkRangeError(double&);
//int findNodeIndex(int*,int,int,int);
//int findNodeIndex(std::vector<int>,int);

inline int checkLogError(double& logVal)
{
 int checkFlag = 0; 
 if (errno == ERANGE) {
  if (logVal < 0) {
   logVal = -100000;
  }
  else {
   checkFlag = -1;
  }
  errno = 0;
 }
 else if (errno == EDOM) {
  logVal = -100000;
  errno = 0;
 }

 return checkFlag;
}

inline int checkRangeError(double& expVal)
{
 int checkFlag = 0; 
 if (errno == ERANGE) {
  if (expVal < 0) {
   expVal = 0;
  }
  else {
   checkFlag = -1;
  }
  errno = 0;
 }

 return checkFlag;
}

inline int findNodeIndex(int* nodePerCliq, int sizCliq, int iCliq, int curNode)
{
 int nodeInd = 0;
 while (nodePerCliq[iCliq*sizCliq + nodeInd] != curNode) {
  ++nodeInd;
 }

 return nodeInd;
}

inline int findNodeIndex(std::vector<int> nodePerCliq, int curNode)
{
 return std::find(nodePerCliq.begin(), nodePerCliq.end(), curNode) - nodePerCliq.begin();
}

double getTime();
double minNonZeroAbsVal(const Eigen::VectorXd &,int);

template<typename T> 
T* addArrayInPlace(T* x, T* y, int sizArray) 
{
 T* firstElem = x;

 for (int i = 0; i != sizArray; ++i) {
  x[i] += y[i];
 }  

 return firstElem;
}

template<typename T>
int addArray(T* x, T* y, T* z, int sizArray) 
{
 for (int i = 0; i != sizArray; ++i) {
  z[i] = x[i] + y[i];
 }

 return 0; 
}

template<typename T>
int addArray(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, int sizArray) 
{
 for (int i = 0; i != sizArray; ++i) {
  z[i] = x[i] + y[i];
 }

 return 0; 
}

template<typename T>
T dotVec(T* x, T* y, int sizArray)
{
 T dotProd = 0;

 for (int i = 0; i != sizArray; ++i) {
  dotProd += x[i]*y[i];
 }

 return dotProd;
}

template<typename T>
T dotVec(std::vector<T> x, std::vector<T> y, int sizArray)
{
 T dotProd = 0;

 for (int i = 0; i != sizArray; ++i) {
  dotProd += x[i]*y[i];
 }

 return dotProd;
}

template<typename T>
T* scaleVectorInPlace(T alpha, T* x, int sizArray)
{
 for (int i = 0; i != sizArray; ++i) {
  x[i] *= alpha;
 }

 return x;
}

template<typename T>
int scaleVector(T alpha, T* x, T* y, int sizArray)
{
 for (int i = 0; i != sizArray; ++i) {
  y[i] = alpha*x[i];
 }

 return 0;
}

template<typename T>
int scaleVector(T alpha, const std::vector<T>& x, std::vector<T>& y, int sizArray)
{
 for (int i = 0; i != sizArray; ++i) {
  y[i] = alpha*x[i];
 }

 return 0;
}


template<typename T>
int argmax(T* array, int lowLim, int upLim)
{
 int maxInd = 0;

 T maxVal = 0;

 for (int i = lowLim; i != upLim; ++i) {
  if (array[i] > maxVal) {
   maxVal = array[i];
   maxInd = i;
  }
 }

 return maxInd;
}

#if 1
template<typename T>
int argmaxAbs(T* array, int lowLim, int upLim)
{
 int maxInd = 0;

 T maxVal = 0;

 for (int i = lowLim; i != upLim; ++i) {
  if (std::abs(array[i]) > maxVal) {
   maxVal = std::abs(array[i]);
   maxInd = i;
  }
 }

 return maxInd;
}

template<typename T>
int argmaxAbs(std::vector<T> array, int lowLim, int upLim)
{
 int maxInd = 0;

 T maxVal = 0;

 for (int i = lowLim; i != upLim; ++i) {
  if (std::abs(array[i]) > maxVal) {
   maxVal = std::abs(array[i]);
   maxInd = i;
  }
 }

 return maxInd;
}
#endif

inline int argmaxAbs(Eigen::VectorXd array, int lowLim, int upLim)
{
 int maxInd = 0;

 double maxVal = 0;

 for (int i = lowLim; i != upLim; ++i) {
  if (std::abs(array(i)) > maxVal) {
   maxVal = std::abs(array(i));
   maxInd = i;
  }
 }

 return maxInd;
}

template<typename T>
double norm(T* vec, int lenVec)
{
 double normVal = 0;

 for (int i = 0; i != lenVec; ++i) {
  normVal += static_cast<double>(vec[i])*(static_cast<double>(vec[i]));
 }

 normVal = sqrt(normVal);

 return normVal;
}

template<typename T>
double norm(std::vector<T> vec, int lenVec)
{
 double normVal = 0;

 for (int i = 0; i != lenVec; ++i) {
  normVal += static_cast<double>(vec[i])*(static_cast<double>(vec[i]));
 }

 normVal = sqrt(normVal);

 return normVal;
}

template<typename T>
inline double l1Norm(const T &ipVec)
{
 double norm = 0;

 int vecSiz = ipVec.size();

 for (int i = 0; i != vecSiz; ++i) {
  norm += std::abs(ipVec[i]);
 }

 return norm;
}

}

#endif
