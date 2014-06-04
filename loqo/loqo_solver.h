// D. Brugger, december 2006
// $Id: loqo_solver.h 573 2010-12-29 10:54:20Z dome $
//
// Copyright (C) 2006 Dominik Brugger
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#ifndef __LOQO__SOLVER__
#define __LOQO__SOLVER__

extern "C" {
  /* #include <PLA.h> */
#include "parallel_loqo.c"
#include "PLA_missing_decls.h"
}
#include "pr_loqo.h"
#include "parallel_loqo.h"
#include "util.h"

class Solver_LOQO : public Solver
{
public:
  Solver_LOQO(int n, int q, int m);
  void Solve(int l, const QMatrix& Q, const double *b, const schar *y,
	     double *alpha, double Cp, double Cn, double eps,
	     SolutionInfo* si, int shrinking);
protected:
  unsigned int NEXT_RAND;
  LOQOfloat *Q_bb;
  LOQOfloat *c;
  LOQOfloat *up;
  LOQOfloat *low;
  LOQOfloat *a;
  LOQOfloat *d;
  LOQOfloat *dist;
  int n; // max. size of working set
  int n_old;
  int m;
  int lmn;
  int iter;
  int q; // # of new variables entering working set each iteration
  char *work_status; // indicating whether variable is in current working set
  int *work_count; // count how many consecutive iterations variable is 
                   // in working set
  LOQOfloat *work_space; // for primal and dual variables of pr_loqo

  double init_margin;
  int init_iter;
  double opt_precision;
  int precision_violations;
  double obj_before;

  enum { WORK_B, WORK_N };
  virtual void setup_a(int *work_set);
  virtual void setup_d(int *not_work_set);
  static const double TOL_ZERO; // tolerance for zero entries
  SolutionInfo *si;
  // working set selection, return 1 if already optimal
  virtual int select_working_set(int *work_set, int *not_work_set); 
  virtual void init_working_set(int *work_set, int *not_work_set);
  void print_problem();
  unsigned int next_rand_pos();
private:
  void setup_up(int *work_set);
  virtual void allocate_a();
  virtual void allocate_d();
  virtual void allocate_work_space();
  // solve the inner qp subproblem
  virtual int solve_inner();
  void setup_low();
  void setup_problem(int *work_set, int *not_work_set);
  // for computation of decision function offset
  //  virtual double calculate_rho() { return work_space[3*n]; }
  // void do_shrinking(); // not implemented yet
};

class Solver_Parallel_LOQO : public Solver_LOQO
{
public:
  Solver_Parallel_LOQO::Solver_Parallel_LOQO(int n, int q, int m, MPI_Comm comm, int nprows, 
					     int npcols, int nb_distr);
  void Solve(int l, const QMatrix& Q, const double *b, const schar *y,
	     double *alpha, double Cp, double Cn, double eps,
	     SolutionInfo* si, int shrinking);
  ~Solver_Parallel_LOQO();
protected:
  virtual void setup_a(int *work_set);
  virtual void setup_d(int *not_work_set);
  void allocate_a();
  void allocate_d();
  enum { CACHED, NOT_CACHED };
  void allocate_work_space();
  int solve_inner(int *work_set);
  void setup_range(int *range_low, int *range_up, int total_sz);
  void setup_up(int *work_set);
  void setup_low();
  void setup_problem(int *work_set, int *not_work_set);
  void sync_gradient(int *work_set, int *not_work_set);
  double calculate_rho() { return dual[0]; }
  // Distributed linear algebra objects.
  PLA_Obj Q_bb_global;
  PLA_Obj c_global;
  PLA_Obj up_global;
  PLA_Obj low_global;
  PLA_Obj a_global;
  PLA_Obj d_global;
  // Space for primal and dual variables of LOQO
  PLA_Obj x; 
  PLA_Obj dist;
  PLA_Obj g; 
  PLA_Obj t; 
  PLA_Obj yy;
  PLA_Obj z;
  PLA_Obj s;
  PLA_Obj plus_one;
  // Views into distributed linear algebra objects
  PLA_Obj Q_bb_global_view;
  PLA_Obj c_global_view;
  PLA_Obj up_global_view;
  PLA_Obj low_global_view;
  PLA_Obj a_global_view;
  PLA_Obj x_view;
  PLA_Obj dist_view;
  PLA_Obj g_view;
  PLA_Obj t_view;
  PLA_Obj z_view;
  PLA_Obj s_view;
  PLA_Obj x_loc_view;
  PLA_Obj dist_loc_view;

  LOQOfloat *alpha_new;
  PLA_Obj x_loc;
  LOQOfloat *dual;
  PLA_Obj yy_loc;
  LOQOfloat *dist_;
  PLA_Obj dist_loc;
  PLA_Template templ;
  // MPI Communicator, rank, group size
  MPI_Comm comm;
  int rank;
  int size;
  int ierr;
  // Local ranges
  int *l_low, *l_up;
  int *n_low, *n_up;
  int *lmn_low, *lmn_up;
  int l_low_loc, l_up_loc;
  int n_low_loc, n_up_loc;
  int lmn_low_loc, lmn_up_loc;
  int local_n;
  int local_lmn;
  int local_l;
  double *G_n;
  int num_rows;
  char *p_cache_status;
};


#endif
