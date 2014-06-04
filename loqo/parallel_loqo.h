// D. Brugger, september 2006
// parallel_loqo.c - parallel implementation of LOQO qp solver using PLAPACK.
// $Id: parallel_loqo.h 573 2010-12-29 10:54:20Z dome $
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

#ifndef __PARALLEL_LOQO_H__
#define __PARALLEL_LOQO_H__

/* Verbosity levels */

#define QUIET 0
#define STATUS 1
#define FLOOD 2

/* Status */

#define STILL_RUNNING               0
#define OPTIMAL_SOLUTION            1
#define SUBOPTIMAL_SOLUTION         2
#define ITERATION_LIMIT             3
#define PRIMAL_INFEASIBLE           4
#define DUAL_INFEASIBLE             5
#define PRIMAL_AND_DUAL_INFEASIBLE  6
#define INCONSISTENT                7
#define PRIMAL_UNBOUNDED            8
#define DUAL_UNBOUNDED              9
#define TIME_LIMIT                  10

/*
 * Solves the quadratic programming problem
 *
 * minimize   c' * x + 1/2 x' * H * x
 * subject to A*x = b
 *            l <= x <= u
 *
 * TODO: Update documentation.
 *
 * verb       : verbosity level
 * sigfig_max : number of significant digits
 * counter_max: stopping criterion
 * restart    : 1 if restart desired
 *
 */

typedef double LOQOfloat;
#define MPIfloat MPI_DOUBLE

#ifdef __cplusplus
extern "C" {
#endif

int parallel_loqo(PLA_Obj c, PLA_Obj h_x, PLA_Obj a, PLA_Obj b, PLA_Obj l,
                  PLA_Obj u, PLA_Obj *x, PLA_Obj *g, PLA_Obj *t, PLA_Obj *y,
                  PLA_Obj *z, PLA_Obj *s, PLA_Obj *dist, int verb,
                  LOQOfloat sigfig_max, int counter_max, LOQOfloat margin,
                  LOQOfloat bound, int restart);

int PLA_Choldc(PLA_Obj a);
void print_matrix(PLA_Obj A);

#ifdef __cplusplus
}
#endif

#endif
