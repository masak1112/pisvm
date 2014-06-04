#ifndef SVM_SOLVER_NU_INCLUDED
#define SVM_SOLVER_NU_INCLUDED
#include "svm_solver.h"

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU : public Solver
{
public:
  Solver_NU() {}
  void Solve(int l, const QMatrix& Q, const double *b, const schar *y,
	     double *alpha, double Cp, double Cn, double eps,
	     SolutionInfo* si, int shrinking);
private:
  SolutionInfo *si;
  int select_working_set(int &i, int &j);
  double calculate_rho();
  void do_shrinking();
};
#endif
