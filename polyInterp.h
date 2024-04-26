#ifndef POLYINTERP_H
#define POLYINTERP_H

//Cubic Backtracking based on Mark Schmidt's minConf code
// Most common case:
// cubic interpolation of 2 points.
// having function and derivative values for both points.
// xminBound and xmaxBound mentioned in Schmidt's code is not applicable.

inline double polyInterp(const std::vector<std::vector<double> > points)
{
 int minPos, notMinPos;

 if (points[0][0] <= points[1][0]) {
  minPos = 0;
  notMinPos = 1;
 }
 else {
  minPos = 1;
  notMinPos = 0;
 }

 double d1 = points[minPos][2] + points[notMinPos][2] - 3*(points[minPos][1] - points[notMinPos][1])/(points[minPos][0] - points[notMinPos][0]);

 double opPos;

 if (pow(d1,2) >= points[minPos][2]*points[notMinPos][2]) {
  double d2 = sqrt(pow(d1,2) - points[minPos][2]*points[notMinPos][2]);

  double t = points[notMinPos][0] - (points[notMinPos][0] - points[minPos][0])*((points[notMinPos][2] + d2 - d1)/(points[notMinPos][2] - points[minPos][2] + 2*d2));

  double tempVal;

  if (t > points[minPos][0]) {
   tempVal = t;
  }
  else {
   tempVal = points[minPos][0];
  }

  if (tempVal < points[notMinPos][0]) {
   opPos = tempVal;
  }
  else {
   opPos = points[notMinPos][0];
  }
 }
 else {
  opPos = 0.5*(points[0][1] + points[1][1]);
 }

 return opPos;
}
#endif // POLYINTERP_H
