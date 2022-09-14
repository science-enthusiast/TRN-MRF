#ifndef TRN_MRF_SOLVERS_SOLVE_SCD_H_
#define TRN_MRF_SOLVERS_SOLVE_SCD_H_

class DualSys;

class SolveSCD {
public:
 static int solve(DualSys*);
private:
 static int updateMSD(DualSys*, int);
 static int updateStar(DualSys*, int); 
}; 

#endif
