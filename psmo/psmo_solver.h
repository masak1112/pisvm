// D. Brugger, december 2006
// $Id: psmo_solver.h 573 2010-12-29 10:54:20Z dome $
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

#ifndef __PSMO__SOLVER__H__
#define __PSMO__SOLVER__H__

#include "util.h"
#define Qmpitype MPI_FLOAT

class Solver_Parallel_SMO : public Solver
{
public:
    Solver_Parallel_SMO(int n, int q, MPI_Comm comm);
    void Solve(int l, const QMatrix& Q, const double *b, const schar *y,
               double *alpha, double Cp, double Cn, double eps,
               SolutionInfo* si, int shrinking);
protected:
    using Solver::select_working_set; // make gcc happy and don't hide overloaded virtual functions
    unsigned int NEXT_RAND;
    // Local storage for subproblems
    Qfloat *Q_bb, *QD_b;
    double *alpha_b, *c;
    schar *a;
    // Max size of working set, etc.
    int n, q, n_old, lmn, iter;
    // Working set status and statistics used by select_working_set
    char *work_status;
    int *work_count;
    int *old_idx;
    enum { WORK_B, WORK_N };
    enum { CACHED, NOT_CACHED };
    static const double TOL_ZERO; // tolerance for zero entries
    SolutionInfo *si;
    virtual int select_working_set(int *work_set, int *not_work_set);
    virtual void init_working_set(int *work_set, int *not_work_set);
    virtual void solve_inner();
    unsigned int next_rand_pos();
private:
    void sync_gradient(int *work_set, int *not_work_set);
    void determine_cached(int *work_set);
    void setup_range(int *range_low, int *range_up, int total_sz);
    // MPI communicator, rank and group size
    MPI_Comm comm;
    int rank, size, ierr;
    // Local ranges of all processors
    int *l_low, *l_up;
    int *n_low, *n_up;
    int *lmn_low, *lmn_up;
    // Local ranges
    int l_low_loc, l_up_loc;
    int n_low_loc, n_up_loc;
    int lmn_low_loc, lmn_up_loc;
    // Local buffer for gradient
    double *G_n;
    // Parallel cache status
    char *p_cache_status;
    int *idx_cached, *idx_not_cached, *nz;
    int count_cached, count_not_cached;
    //MPI Types for transmitting Q_bb
    void setup_Q_bb_mpitypes() {
        const int upper_half = n-(n/(2*size))*size;
        if(mpitypes_n == 0) {
            gatherv_Qbb_unsplittable_displ = new int[size];
            gatherv_Qbb_unsplittable_counts = new int[size];
        }
        if (n%size != 0) {
            //First rows that are not distributed equally (rows 0 to n%size - NOT dividable by 'size')
            //last processor did calculate row (n%size) - 1

            const int unused = size-(n%size);
            for (int i = 0;i < unused; ++i) {
                gatherv_Qbb_unsplittable_displ[i] = 0;
                gatherv_Qbb_unsplittable_counts[i] = 0;
            }
            for (int i = unused; i < size; ++i) {
                gatherv_Qbb_unsplittable_displ[i] = (i - unused)*n;
                gatherv_Qbb_unsplittable_counts[i] = i - unused + 1;
            }
        }


        int *indexed_cnt;
        int *indexed_displ;
        {
             //Upper half - rows (n%size) + 1 to upper_half (amount is dividable by 'size': floor(n/2p)*2p + (n%2p > p ? p : 0)
            if (mpitypes_n != 0) {
                MPI_Type_free(&mpitype_Qbb_upper_rows_resized);
                MPI_Type_free(&mpitype_Qbb_upper_rows);
            }
            const int rowsPerProcess = n/(2*size) + (n%(2*size) >= size ? 1 : 0);
            indexed_cnt = new int[rowsPerProcess];
            indexed_displ = new int[rowsPerProcess];
            for (int i = 0; i < rowsPerProcess; i++) {
                indexed_cnt[i] = (n%size) + (i+1) * size;
                indexed_displ[i] = i*n*size;
            }
            MPI_Type_indexed(rowsPerProcess,indexed_cnt,indexed_displ,Qmpitype,&mpitype_Qbb_upper_rows);
            MPI_Type_commit(&mpitype_Qbb_upper_rows);
            MPI_Type_create_resized(mpitype_Qbb_upper_rows,0,n*sizeof(Qfloat),&mpitype_Qbb_upper_rows_resized);
            MPI_Type_commit(&mpitype_Qbb_upper_rows_resized);
        }
        if (mpitypes_n == 0) {
            gatherv_Qbb_lower_counts = new int[size];
            gatherv_Qbb_lower_displ = new int[size];
        }
        if (n >= 2 * size) {
            //There is a lower half
            //rows upper_half to n (dividable by 'size')
            if (mpitypes_n >= 2 * size) {
                //There are old types saved - free them
                MPI_Type_free(&mpitype_Qbb_lower_rows_resized);
                MPI_Type_free(&mpitype_Qbb_lower_rows);
            }
            //There is a lower half
            //rows upper_half to n (dividable by 'size')
            const int rowsPerProcess = n/(2*size);
            for (int i = 0; i < rowsPerProcess; i++) {
                indexed_cnt[i] = upper_half + (i+1) * size;
                indexed_displ[i] = i*n*size;
            }
            MPI_Type_indexed(rowsPerProcess,indexed_cnt,indexed_displ,Qmpitype,&mpitype_Qbb_lower_rows);
            MPI_Type_commit(&mpitype_Qbb_lower_rows);
            MPI_Type_create_resized(mpitype_Qbb_lower_rows,0,n*sizeof(Qfloat),&mpitype_Qbb_lower_rows_resized);
            MPI_Type_commit(&mpitype_Qbb_lower_rows_resized);

            for (int i = 0; i < size; i++) {
                gatherv_Qbb_lower_counts[i] = 1;
                gatherv_Qbb_lower_displ[i] = (size - i - 1);
            }
        } else if (mpitypes_n >= 2 * size) {
            //We used them before, but don't need them anymore - free them.
            MPI_Type_free(&mpitype_Qbb_lower_rows_resized);
            MPI_Type_free(&mpitype_Qbb_lower_rows);
        }

        delete[] indexed_cnt;
        delete[] indexed_displ;

        mpitypes_n = n;
    }
    int mpitypes_n;
    MPI_Datatype mpitype_Qbb_upper_rows;
    MPI_Datatype mpitype_Qbb_upper_rows_resized;
    MPI_Datatype mpitype_Qbb_lower_rows;
    MPI_Datatype mpitype_Qbb_lower_rows_resized;
    int *gatherv_Qbb_unsplittable_counts;
    int *gatherv_Qbb_unsplittable_displ;
    int *gatherv_Qbb_lower_counts;
    int *gatherv_Qbb_lower_displ;
};

class Solver_Parallel_SMO_NU : public Solver_Parallel_SMO
{
public:
    Solver_Parallel_SMO_NU(int n, int q, MPI_Comm comm);
protected:
    void solve_inner();
    int select_working_set(int *work_set, int *not_work_set);
    double calculate_rho();
};

#endif
