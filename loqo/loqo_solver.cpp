// D. Brugger, december 2006
// $Id: loqo_solver.cpp 573 2010-12-29 10:54:20Z dome $
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

#include "loqo_solver.h"

#ifndef CheckError
#define CheckError(n) if(n){printf("line %d, file %s\n",__LINE__,__FILE__);}
#endif

const double Solver_LOQO::TOL_ZERO = 1e-06; // tolerance for zero entries

Solver_LOQO::Solver_LOQO(int n, int q, int m)
{
    // Ensure that n,q are even numbers.
    this->n = n % 2 == 0 ? n : n-1;
    this->n_old = this->n;
    this->q = q % 2 == 0 ? q : q-1;
    this->m = m;
    // Ensure sane q
    this->q = this->q > this->n ? this->n : this->q;
    this->init_margin = 0.1;
    this->init_iter = 500;
    this->precision_violations = 0;
    this->opt_precision = 1e-8;
    NEXT_RAND = 1;
}

unsigned int Solver_LOQO::next_rand_pos()
{
    NEXT_RAND = NEXT_RAND*1103515245L + 12345L;
    return NEXT_RAND & 0x7fffffff;
}

void Solver_LOQO::setup_up(int *work_set)
{
    for(int i=0; i<n; ++i)
        up[i] = get_C(work_set[i]);
}
void Solver_LOQO::allocate_a()
{
    a = new double[n];
}
void Solver_LOQO::allocate_d()
{
    d = new double[1];
}
void Solver_LOQO::allocate_work_space() {
    work_space = new double[5*n+1];
}

void Solver_LOQO::setup_low()
{
    for(int i=0; i<n; ++i)
        low[i] = 0;
}

Solver_Parallel_LOQO::Solver_Parallel_LOQO(int n, int q, int m, MPI_Comm comm, int nprows,
        int npcols, int nb_distr) : Solver_LOQO(n,q,m)
{
    this->comm = comm;
    ierr = MPI_Comm_rank(comm, &this->rank);
    CheckError(ierr);
    ierr = MPI_Comm_size(comm, &this->size);
    CheckError(ierr);
    ierr = PLA_Comm_1D_to_2D(comm, nprows, npcols, &comm);
    CheckError(ierr);
    // Initialize PLAPACK
    ierr = PLA_Init(comm);
    CheckError(ierr);
    ierr = PLA_Temp_create(nb_distr, 0, &templ);
    CheckError(ierr);
    // Initialize PLAPACK objs
    Q_bb_global = NULL;
    c_global = NULL;
    up_global = NULL;
    low_global = NULL;
    a_global = NULL;
    d_global = NULL;
    x = NULL;
    dist = NULL;
    g = NULL;
    t = NULL;
    yy = NULL;
    z = NULL;
    s = NULL;
    plus_one = NULL;
    Q_bb_global_view = NULL;
    c_global_view = NULL;
    up_global_view = NULL;
    low_global_view = NULL;
    a_global_view = NULL;
    x_view = NULL;
    dist_view = NULL;
    g_view = NULL;
    t_view = NULL;
    z_view = NULL;
    s_view = NULL;
    x_loc_view = NULL;
    x_loc = NULL;
    yy_loc = NULL;
    dist_loc = NULL;
    dist_loc_view = NULL;
}

Solver_Parallel_LOQO::~Solver_Parallel_LOQO()
{
    // Finalize PLAPACK
    ierr = PLA_Finalize();
    CheckError(ierr);
}

class Solver_Parallel_LOQO_NU : public Solver_Parallel_LOQO
{
public:
    Solver_Parallel_LOQO_NU(int n, int q, int m, MPI_Comm comm, int nprows,
                            int npcols, int nb_distr, double nu)
        : Solver_Parallel_LOQO(n, q, m, comm, nprows, npcols, nb_distr)
    {
        this->nu = nu;
    };
private:
    double nu;
    void setup_a(int *work_set);
    void setup_d(int *not_work_set);
    int select_working_set(int *work_set, int *not_work_set);
//   double calculate_rho() { si->r = dual[1]; return dual[0]; }
    double calculate_rho();
};

double Solver_Parallel_LOQO_NU::calculate_rho() {
    printf("Solver_Parallel_LOQO_NU::calculate_rho called!\n");
    int nr_free1 = 0,nr_free2 = 0;
    double ub1 = INF, ub2 = INF;
    double lb1 = -INF, lb2 = -INF;
    double sum_free1 = 0, sum_free2 = 0;

//   printf("alpha = ");
//   for(int i=0; i<l; ++i)
//     printf(" %g",alpha[i]);
//   printf("\n");

    for(int i=0; i<l; i++)
    {
        if(y[i]==+1)
        {
            if(is_lower_bound(i))
                ub1 = min(ub1,G[i]);
            else if(is_upper_bound(i))
                lb1 = max(lb1,G[i]);
            else
            {
                ++nr_free1;
                sum_free1 += G[i];
            }
        }
        else
        {
            if(is_lower_bound(i))
                ub2 = min(ub2,G[i]);
            else if(is_upper_bound(i))
                lb2 = max(lb2,G[i]);
            else
            {
                ++nr_free2;
                sum_free2 += G[i];
            }
        }
    }
    printf("nr_free1 = %d\n", nr_free1);
    printf("sum_free1 = %g\n",sum_free1);
    printf("nr_free2 = %d\n", nr_free2);
    printf("sum_freee = %g\n",sum_free2);
    double r1,r2;
    if(nr_free1 > 0)
        r1 = sum_free1/nr_free1;
    else
        r1 = (ub1+lb1)/2;

    if(nr_free2 > 0)
        r2 = sum_free2/nr_free2;
    else
        r2 = (ub2+lb2)/2;

    si->r = (r1+r2)/2;
    printf("(r1+r2)/2 = %g\n", (r1+r2)/2);
    printf("(r1+r2)/2 = %g\n", (r1-r2)/2);
    return (r1-r2)/2;
}

void Solver_Parallel_LOQO_NU::setup_a(int *work_set)
{
    info("Solver_Parallel_LOQO_NU::setup_a called\n");
    // Note that a has to be in column major layout.
    for(int i=n_low_loc; i<n_up_loc; ++i)
    {
        a[(i-n_low_loc)*m] = y[work_set[i]];
        a[(i-n_low_loc)*m+1] = 1;
    }
    ierr = PLA_Obj_set_to_zero(a_global_view);
    CheckError(ierr);
    ierr = PLA_API_begin();
    CheckError(ierr);
    ierr = PLA_Obj_API_open(a_global_view);
    CheckError(ierr);
    ierr = PLA_API_axpy_matrix_to_global(m, local_n, plus_one, (void *)a, m,
                                         a_global_view, 0, n_low_loc);
    CheckError(ierr);
    ierr = PLA_Obj_API_close(a_global_view);
    CheckError(ierr);
    ierr = PLA_API_end();
    CheckError(ierr);
    info("Solver_Parallel_LOQO_NU::setup_a done\n");
}

void Solver_Parallel_LOQO_NU::setup_d(int *not_work_set)
{
    info("Solver_Parallel_LOQO_NU::setup_d called\n");
    d[0] = 0;
    for(int i=0; i<lmn; ++i)
    {
        if(fabs(alpha[not_work_set[i]]) > TOL_ZERO)
            d[0] -= y[not_work_set[i]]*alpha[not_work_set[i]];
    }
    d[1] = nu*l;
    info("Setting d[1] = %g\n", d[1]);
    info_flush();
    for(int i=0; i<lmn; ++i)
        d[1] -= alpha[not_work_set[i]];
    ierr = PLA_Obj_set_to_zero(d_global);
    CheckError(ierr);
    ierr = PLA_API_begin();
    CheckError(ierr);
    ierr = PLA_Obj_API_open(d_global);
    CheckError(ierr);
    if(rank == 0)
        ierr = PLA_API_axpy_vector_to_global(m, plus_one, (void *) d, 1, d_global,
                                             0);
    CheckError(ierr);
    ierr = PLA_Obj_API_close(d_global);
    CheckError(ierr);
    ierr = PLA_API_end();
    CheckError(ierr);
    info("Solver_Parallel_LOQO_NU::setup_d done\n");
    info_flush();
}

void Solver_Parallel_LOQO::setup_range(int *range_low, int *range_up,
                                       int total_sz)
{
    int local_sz = total_sz/size;
    int idx_up = local_sz;
    int idx_low = 0;
    if(total_sz != 0)
    {
        for(int i=0; i<size-1; ++i)
        {
            range_low[i] = idx_low;
            range_up[i] = idx_up;
            idx_low = idx_up;
            idx_up = idx_low + local_sz + 1;
        }
        range_low[size-1] = idx_low;
        range_up[size-1]=total_sz;
    }
    else
    {
        for(int i=0; i<size; ++i)
        {
            range_low[i] = 0;
            range_up[i] = 0;
        }
    }
}

int Solver_Parallel_LOQO_NU::select_working_set(int *work_set,
        int *not_work_set)
{
    printf("selecting working set...");
    // reset work status
    n = n_old;
    for(int i=0; i<l; ++i)
    {
        work_status[i] = WORK_N;
    }
    double Gmin1 = INF;
    double Gmin2 = INF;
    double Gmax1 = -INF;
    double Gmax2 = -INF;
    int min1 = -1;
    int min2 = -1;
    int max1 = -1;
    int max2 = -1;
    for(int t=0; t<l; ++t)
    {
        if(y[t] == +1)
        {
            if(!is_upper_bound(t))
            {
                if(G[t] < Gmin1)
                {
                    Gmin1 = G[t];
                    min1 = t;
                }
            }
            if(!is_lower_bound(t))
            {
                if(G[t] > Gmax1)
                {
                    Gmax1 = G[t];
                    max1 = t;
                }
            }
        }
        else
        {
            if(!is_upper_bound(t))
            {
                if(G[t] < Gmin2)
                {
                    Gmin2 = G[t];
                    min2 = t;
                }
            }
            if(!is_lower_bound(t))
            {
                if(G[t] > Gmax2)
                {
                    Gmax2 = G[t];
                    max2 = t;
                }
            }
        }
    }
    // check for optimality, max. violating pair.
    printf("max(Gmax1-Gmin1,Gmax2-Gmin2) = %g < %g\n",
           max(Gmax1-Gmin1,Gmax2-Gmin2),eps);
    if(max(Gmax1-Gmin1,Gmax2-Gmin2) < eps)
        return 1;

    // Sort gradient
    double *Gtmp = new double[l];
    int *pidx = new int[l];
    for(int i=0; i<l; ++i)
    {
        Gtmp[i] = G[i];
        pidx[i] = i;
    }
    quick_sort(Gtmp, pidx, 0, l-1);
//   printf("pidx = ");
//   for(int i=0; i<l; ++i)
//     printf(" %d", pidx[i]);
//   printf("\n");

    int top1=l-1;
    int top2=l-1;
    int bot1=0;
    int bot2=0;
    int count=0;
    // Select a full set initially
    int nselect = iter == 0 ? n : q;
    while(count < nselect)
    {
        if(top1 > bot1)
        {
            while(!( (is_free(pidx[top1]) || is_upper_bound(pidx[top1]))
                     && y[pidx[top1]] == +1))
            {
                if(top1 <= bot1) break;
                --top1;
            }
            while(!( (is_free(pidx[bot1]) || is_lower_bound(pidx[bot1]))
                     && y[pidx[bot1]] == +1))
            {
                if(bot1 >= top1) break;
                ++bot1;
            }
        }
        if(top2 > bot2)
        {
            while(!( (is_free(pidx[top2]) || is_upper_bound(pidx[top2]))
                     && y[pidx[top2]] == -1))
            {
                if(top2 <= bot2) break;
                --top2;
            }
            while(!( (is_free(pidx[bot2]) || is_lower_bound(pidx[bot2]))
                     && y[pidx[bot2]] == -1))
            {
                if(bot2 >= top2) break;
                ++bot2;
            }
        }
        if(top1 > bot1 && top2 > bot2)
        {
            if(G[pidx[top1]]-G[pidx[bot1]] > G[pidx[top2]]-G[pidx[bot2]])
            {
                work_status[pidx[top1]] = WORK_B;
                work_status[pidx[bot1]] = WORK_B;
                --top1;
                ++bot1;
            }
            else
            {
                work_status[pidx[top2]] = WORK_B;
                work_status[pidx[bot2]] = WORK_B;
                --top2;
                ++bot2;
            }
            count += 2;
        }
        else if(top1 > bot1)
        {
            work_status[pidx[top1]] = WORK_B;
            work_status[pidx[bot1]] = WORK_B;
            --top1;
            ++bot1;
            count += 2;
        }
        else if(top2 > bot2)
        {
            work_status[pidx[top2]] = WORK_B;
            work_status[pidx[bot2]] = WORK_B;
            --top2;
            ++bot2;
            count += 2;
        }
        else
            break;
    } // while(count < nselect)
    if(count < n)
    {
        // Compute subset of indices in previous working set
        // which were not yet selected
        int j=0;
        int *work_count_subset = new int[l-count];
        int *subset = new int[l-count];
        int *psubset = new int[l-count];
        for(int i=0; i<l; ++i)
        {
            if(work_status[i] == WORK_N && work_count[i] > -1)
            {
                work_count_subset[j] = work_count[i];
                subset[j] = i;
                psubset[j] = j;
                ++j;
            }
        }
        quick_sort(work_count_subset, psubset, 0, j-1);

        // Fill up with j \in B, 0 < alpha[j] < C
        for(int i=0; i<j; ++i)
        {
            if(count == n) break;
            if(is_free(subset[psubset[i]]))
            {
                work_status[subset[psubset[i]]] = WORK_B;
                ++count;
            }
        }
        // Fill up with j \in B, alpha[j] = 0
        for(int i=0; i<j; ++i)
        {
            if(count == n) break;
            if(is_lower_bound(subset[psubset[i]]))
            {
                work_status[subset[psubset[i]]] = WORK_B;
                ++count;
            }
        }
        // Fill up with j \in B, alpha[j] = C
        for(int i=0; i<j; ++i)
        {
            if(count == n) break;
            if(is_upper_bound(subset[psubset[i]]))
            {
                work_status[subset[psubset[i]]] = WORK_B;
                ++count;
            }
        }
        // clean up
        delete[] work_count_subset;
        delete[] subset;
        delete[] psubset;
    } // if(count < n)
    // Setup work_set and not_work_set
    // update work_count
    int nnew=0;
    int i=0;
    int j=0;
    n=0;
    for(int t=0; t<l; ++t)
    {
        if(work_status[t] == WORK_B)
        {
            if(work_count[t] == -1)
                ++nnew;
            work_set[i] = t;
            ++i;
            ++n;
            ++work_count[t];
        }
        else
        {
            not_work_set[j] = t;
            ++j;
            work_count[t] = -1;
        }
    }
    // Update q
    printf("nnew = %d\n", nnew);
    int kin = nnew;
    nnew = nnew % 2 == 0 ? nnew : nnew-1;
    int L = n/10 % 2 == 0 ? n/10 : (n/10)-1;
    q = min(q, max( max( 10, L ), nnew ) );
    printf("q = %d\n", q);
    printf("n = %d\n",n);
    if(kin == 0)
    {
        // Increase precision of solver.
        if(opt_precision > 1e-20)
            opt_precision /= 100;
        else
        {
            info("Error: Unable to select a suitable working set!!!\n");
            return 1;
        }
    }
    // clean up
    delete[] Gtmp;
    delete[] pidx;
    printf("done.\n");
    return 0;
}


void Solver_Parallel_LOQO::allocate_a()
{
    a = new LOQOfloat[m*local_n];
    ierr = PLA_Matrix_create(MPIfloat, m, n, templ, PLA_ALIGN_FIRST,
                             PLA_ALIGN_FIRST, &a_global);
    CheckError(ierr);
}

void Solver_Parallel_LOQO::allocate_d()
{
    d = new LOQOfloat[m];
    ierr = PLA_Mvector_create(MPIfloat, m, 1, templ, PLA_ALIGN_FIRST,
                              &d_global);
    CheckError(ierr);
}

void Solver_Parallel_LOQO::allocate_work_space()
{
    ierr = PLA_Mvector_create(MPIfloat, n, 1, templ, PLA_ALIGN_FIRST, &x);
    CheckError(ierr);
    ierr = PLA_Mvector_create(MPIfloat, n, 1, templ, PLA_ALIGN_FIRST, &dist);
    CheckError(ierr);
    ierr = PLA_Mvector_create(MPIfloat, n, 1, templ, PLA_ALIGN_FIRST, &g);
    CheckError(ierr);
    ierr = PLA_Mvector_create(MPIfloat, n, 1, templ, PLA_ALIGN_FIRST, &t);
    CheckError(ierr);
    ierr = PLA_Mvector_create(MPIfloat, m, 1, templ, PLA_ALIGN_FIRST, &yy);
    CheckError(ierr);
    ierr = PLA_Mvector_create(MPIfloat, n, 1, templ, PLA_ALIGN_FIRST, &z);
    CheckError(ierr);
    ierr = PLA_Mvector_create(MPIfloat, n, 1, templ, PLA_ALIGN_FIRST, &s);
    CheckError(ierr);
}

void Solver_Parallel_LOQO::setup_a(int *work_set)
{
    for(int i=n_low_loc; i<n_up_loc; ++i)
        a[i-n_low_loc] = y[work_set[i]];
    ierr = PLA_Obj_set_to_zero(a_global_view);
    CheckError(ierr);
    ierr = PLA_API_begin();
    CheckError(ierr);
    ierr = PLA_Obj_API_open(a_global_view);
    CheckError(ierr);
    ierr = PLA_API_axpy_matrix_to_global(1, local_n, plus_one, (void *)a, 1,
                                         a_global_view, 0, n_low_loc);
    CheckError(ierr);
    ierr = PLA_Obj_API_close(a_global_view);
    CheckError(ierr);
    ierr = PLA_API_end();
    CheckError(ierr);
}

void Solver_Parallel_LOQO::setup_d(int *not_work_set)
{
    d[0] = 0;
    for(int i=0; i<lmn; ++i)
    {
        if(fabs(alpha[not_work_set[i]]) > TOL_ZERO)
            d[0] -= y[not_work_set[i]]*alpha[not_work_set[i]];
    }
    ierr = PLA_Obj_set_to_zero(d_global);
    CheckError(ierr);
    ierr = PLA_API_begin();
    CheckError(ierr);
    ierr = PLA_Obj_API_open(d_global);
    CheckError(ierr);
    if(rank == 0)
        ierr = PLA_API_axpy_vector_to_global(1, plus_one, (void *) d, 1, d_global,
                                             0);
    CheckError(ierr);
    ierr = PLA_Obj_API_close(d_global);
    CheckError(ierr);
    ierr = PLA_API_end();
    CheckError(ierr);
}

void Solver_Parallel_LOQO::setup_up(int *work_set)
{
    for(int i=n_low_loc; i<n_up_loc; ++i)
        up[i-n_low_loc] = get_C(work_set[i]);
    ierr = PLA_Obj_set_to_zero(up_global_view);
    CheckError(ierr);
    ierr = PLA_API_begin();
    CheckError(ierr);
    ierr = PLA_Obj_API_open(up_global_view);
    CheckError(ierr);
    ierr = PLA_API_axpy_vector_to_global(local_n, plus_one, (void *) up, 1,
                                         up_global_view, n_low_loc);
    CheckError(ierr);
    ierr = PLA_Obj_API_close(up_global_view);
    CheckError(ierr);
    ierr = PLA_API_end();
    CheckError(ierr);
}

void Solver_Parallel_LOQO::setup_low()
{
    for(int i=n_low_loc; i<n_up_loc; ++i)
        low[i-n_low_loc] = 0;
    ierr = PLA_Obj_set_to_zero(low_global_view);
    CheckError(ierr);
    ierr = PLA_API_begin();
    CheckError(ierr);
    ierr = PLA_Obj_API_open(low_global_view);
    CheckError(ierr);
    ierr = PLA_API_axpy_vector_to_global(local_n, plus_one, (void *) low, 1,
                                         low_global_view, n_low_loc);
    CheckError(ierr);
    ierr = PLA_Obj_API_close(low_global_view);
    CheckError(ierr);
    ierr = PLA_API_end();
    CheckError(ierr);
}

void Solver_Parallel_LOQO::setup_problem(int *work_set, int *not_work_set)
{
    // Note: We have to store Q_bb in column major format
    // since this is storage layout expected by PLAPACK-API.
    int start_i = 0;
    for(int i=0; i<n; ++i, ++start_i)
    {
        if(work_set[i] >= l_low_loc)
            break;
    }
    num_rows = 0;
    for(int i=start_i; work_set[i] < l_up_loc && i<n; ++i)
        ++num_rows;
    for(int i=start_i; work_set[i] < l_up_loc && i<n; ++i)
        //for(int i=n_low_loc; i<n_up_loc; ++i)
    {
        const Qfloat *Q_i = Q->get_Q_subset(work_set[i], work_set, n);
        //c[i-n_low_loc] = G[work_set[i]];
        c[i-start_i] = G[work_set[i]];
        for(int j=0; j<n; ++j)
        {
            //Q_bb[j*local_n+(i-n_low_loc)] = (LOQOfloat) Q_i[work_set[j]];
            Q_bb[j*num_rows+(i-start_i)] = (LOQOfloat) Q_i[work_set[j]];
            if(alpha[work_set[j]] > TOL_ZERO)
                // c[i-n_low_loc] -= Q_bb[j*local_n+(i-n_low_loc)]*alpha[work_set[j]];
                c[i-start_i] -= Q_bb[j*num_rows+(i-start_i)]*alpha[work_set[j]];

        }
    }
    ierr = MPI_Barrier(comm);
    CheckError(ierr);

    // Setup distributed objects.
    ierr = PLA_Obj_set_to_zero(Q_bb_global_view);
    CheckError(ierr);
    ierr = PLA_Obj_set_to_zero(c_global_view);
    CheckError(ierr);
    ierr = PLA_API_begin();
    CheckError(ierr);
    ierr = PLA_Obj_API_open(Q_bb_global_view);
    CheckError(ierr);
    ierr = PLA_Obj_API_open(c_global_view);
    CheckError(ierr);
//   ierr = PLA_API_axpy_matrix_to_global(local_n, n, plus_one, (void *) Q_bb,
// 				       local_n, Q_bb_global_view, n_low_loc,
// 				       0); CheckError(ierr);
    ierr = PLA_API_axpy_matrix_to_global(num_rows, n, plus_one, (void *) Q_bb,
                                         num_rows, Q_bb_global_view, start_i,
                                         0);
    CheckError(ierr);
//   ierr = PLA_API_axpy_vector_to_global(local_n, plus_one, (void *) c, 1,
// 				       c_global_view, n_low_loc);
//   CheckError(ierr);
    ierr = PLA_API_axpy_vector_to_global(num_rows, plus_one, (void *) c, 1,
                                         c_global_view, start_i);
    CheckError(ierr);
    ierr = PLA_Obj_API_close(Q_bb_global_view);
    CheckError(ierr);
    ierr = PLA_Obj_API_close(c_global_view);
    CheckError(ierr);
    ierr = PLA_API_end();
    CheckError(ierr);
    setup_a(work_set);
    setup_d(not_work_set);
}

int Solver_Parallel_LOQO::solve_inner(int *work_set)
{
    int result = -1;
    double sigdig;
    double epsilon_loqo = 1e-10;
    double margin;
    int iteration;
    for(margin=init_margin, iteration=init_iter;
            margin <= 0.9999999 && result != OPTIMAL_SOLUTION;)
    {
        sigdig = -log10(opt_precision);
        // run pr_loqo
        result = parallel_loqo(c_global_view, Q_bb_global_view, a_global_view,
                               d_global, low_global_view, up_global_view,
                               &x_view, &g_view, &t_view,
                               &yy, &z_view, &s_view, &dist_view, 3,
                               sigdig, iteration, margin, up[0]/4, 0);
        // Make solution known to all processors, local access
        // occurs via alpha_new dist_ and dual
        ierr = PLA_Copy(x_view, x_loc_view);
        CheckError(ierr);
        ierr = PLA_Copy(dist_view, dist_loc_view);
        CheckError(ierr);
        ierr = PLA_Copy(yy, yy_loc);
        CheckError(ierr);
        // TODO: What happens if we have a choldc problem in the parallel
        // version? Should we implement Manteuffel shifting or find a
        // workaround here?
        if(result != OPTIMAL_SOLUTION)
        {
            // increase number of iterations
            iteration += 2000;
            init_iter += 10;
            // reduce precision
            opt_precision *= 10.0;
            info("NOTICE: Reducing precision of PR_LOQO!\n");
            info_flush();
        }
    }
    // Check precision of alphas
    for(int i=0; i<n; ++i)
    {
        // TODO: Should use up[i] and low[i], but they are not global
        if(alpha_new[i] < get_C(i)-epsilon_loqo && dist_[i] < -eps)
            epsilon_loqo = 2*(get_C(i)-alpha_new[i]);
        else if(alpha_new[i] > 0+epsilon_loqo && dist_[i] > eps)
            epsilon_loqo = 2*alpha_new[i];
    }
    info("Using epsilon_loqo= %g\n", epsilon_loqo);
    info_flush();
    // Clip alphas to bounds
    for(int i=0; i<n; ++i)
    {
        if(fabs(alpha_new[i]) < epsilon_loqo)
        {
            alpha_new[i] = 0;
        }
        if(fabs(alpha_new[i]-get_C(i)) < epsilon_loqo)
        {
            alpha_new[i] = get_C(i);
        }
    }
    // Compute obj after optimization
    double obj_after = 0.0;
    int start_i = 0;
    for(int i=0; i<n; ++i, ++start_i)
    {
        if(work_set[i] >= l_low_loc)
            break;
    }
    for(int i=start_i; work_set[i] < l_up_loc && i<n; ++i)
//   for(int i=n_low_loc; i<n_up_loc; ++i)
    {
        //      obj_after += alpha_new[i]*c[i-n_low_loc];
        obj_after += alpha_new[i]*c[i-start_i];
        for(int j=0; j<n; ++j)
// 	obj_after += 0.5*alpha_new[i]*Q_bb[j*local_n+(i-n_low_loc)]
// 	  *alpha_new[j];
            obj_after += 0.5*alpha_new[i]*Q_bb[j*num_rows+(i-start_i)]
                         *alpha_new[j];
    }
    // Get local changes for obj_after from other processors
    double obj_after_loc=0.0;
    double obj_after_other=0.0;
    for(int i=0; i<size; ++i)
    {
        if(i == rank)
            ierr = MPI_Bcast(&obj_after, 1, MPI_DOUBLE, i, comm);
        else
        {
            ierr = MPI_Bcast(&obj_after_loc, 1, MPI_DOUBLE, i, comm);
            obj_after_other += obj_after_loc;
        }
    }
    obj_after += obj_after_other;
    printf("obj after/before = %g / %g\n", obj_after, obj_before);
    fflush(stdout);
    printf("delta a/b = %g\n", obj_after-obj_before);
    fflush(stdout);

    // Check for progress
    if(obj_after >= obj_before)
    {
        // Increase precision
        opt_precision /= 100.0;
        ++precision_violations;
        info("NOTICE: Increasing Precision of PR_LOQO.\n");
        info_flush();
    }
    if(precision_violations > 500)
    {
        // Relax stopping criterion
        eps *= 10.0;
        precision_violations = 0;
        info("WARNING: Relaxing epsilon on KKT-Conditions.\n");
        info_flush();
    }
    return result;
}

void Solver_Parallel_LOQO::Solve(int l, const QMatrix& Q, const double *b_,
                                 const schar *y_, double *alpha_, double Cp,
                                 double Cn, double eps, SolutionInfo* si,
                                 int shrinking)
{
    double total_time = MPI_Wtime();
    double problem_setup_time = 0;
    double inner_solver_time = 0;
    double gradient_updating_time = 0;
    double time;
    // Initialization
    this->l = l;
    this->Q = &Q;
    this->si = si;
    clone(b,b_,l);
    clone(y,y_,l);
    clone(alpha,alpha_,l);
    this->Cp = Cp;
    this->Cn = Cn;
    this->eps = eps;
    this->lmn = l - n;
    int *work_set = new int[n];
    int *not_work_set = new int[l];
    double *delta_alpha = new double[n];
    this->G_n = new double[lmn];
    this->p_cache_status = new char[size*l];
    // Setup alpha status and work status
    {
        alpha_status = new char[l];
        work_status = new char[l];
        work_count = new int[l];
        for(int i=0; i<l; ++i)
        {
            update_alpha_status(i);
            work_status[i] = WORK_N;
            work_count[i] = -1;
            for(int k=0; k<size; ++k)
                p_cache_status[k*size+i] = NOT_CACHED;
        }
    }
    // Setup local index ranges
    this->l_low = new int[size];
    this->l_up = new int[size];
    this->n_low = new int[size];
    this->n_up = new int[size];
    this->lmn_low = new int[size];
    this->lmn_up = new int[size];
    setup_range(l_low, l_up, l);
    setup_range(n_low, n_up, n);
    setup_range(lmn_low, lmn_up, lmn);
    this->l_low_loc = l_low[rank];
    this->l_up_loc = l_up[rank];
    this->n_low_loc = n_low[rank];
    this->n_up_loc = n_up[rank];
    this->lmn_up_loc = lmn_up[rank];
    this->lmn_low_loc = lmn_low[rank];
    this->local_l = l_up_loc - l_low_loc;
    this->local_n = n_up_loc - n_low_loc;
    this->local_lmn = lmn_up_loc - lmn_low_loc;

    // Setup gradient
    {
        info("Initializing gradient...");
        info_flush();
        G = new double[l];
        double *G_send = new double[l];
        double *G_recv = new double[l];
        for(int i=0; i<l; ++i)
        {
            G[i] = b[i];
            G_send[i] = 0;
        }
        // Compute local portion of gradient
        for(int i=l_low_loc; i<l_up_loc; ++i)
        {
            if(!is_lower_bound(i))
            {
                const Qfloat *Q_i = Q.get_Q(i,l);
                double alpha_i = alpha[i];
                for(int j=0; j<l; ++j)
                    G_send[j] += alpha_i * Q_i[j];
            }
        }
        // Get contributions from other processors
        for(int k=0; k<size; ++k)
        {
            if(rank == k)
            {
                ierr = MPI_Bcast(G_send, l, MPIfloat, k, comm);
                CheckError(ierr);
                for(int i=0; i<l; ++i)
                    G[i] += G_send[i];
            }
            else
            {
                ierr = MPI_Bcast(G_recv, l, MPIfloat, k, comm);
                CheckError(ierr);
                for(int i=0; i<l; ++i)
                    G[i] += G_recv[i];
            }
        }
        delete[] G_recv;
        delete[] G_send;

//     for(int i=0; i<l; ++i)
//       {
//  	if(!is_lower_bound(i))
//  	  {
// 	    const Qfloat *Q_i = Q.get_Q(i,l);
// 	    double alpha_i = alpha[i];
// 	    for(int j=0; j<l; ++j)
// 	      {
// 		G[j] += alpha_i * Q_i[j];
// 	      }
//  	  }
//       }
        info("done.\n");
        info_flush();
    }
    // Allocate local & global space for problem setup
    //Q_bb = new LOQOfloat[local_n * n];
    Q_bb = new LOQOfloat[n*n];
    //   c = new LOQOfloat[local_n];
    c = new LOQOfloat[n];
    up = new LOQOfloat[local_n];
    low = new LOQOfloat[local_n];
    allocate_a();
    allocate_d();
    allocate_work_space();
    ierr = PLA_Matrix_create(MPIfloat, n, n, templ, PLA_ALIGN_FIRST,
                             PLA_ALIGN_FIRST, &Q_bb_global);
    CheckError(ierr);
    ierr = PLA_Mvector_create(MPIfloat, n, 1, templ, PLA_ALIGN_FIRST, &c_global);
    CheckError(ierr);
    ierr = PLA_Mvector_create(MPIfloat, n, 1, templ, PLA_ALIGN_FIRST, &up_global);
    CheckError(ierr);
    ierr = PLA_Mvector_create(MPIfloat, n, 1, templ, PLA_ALIGN_FIRST, &low_global);
    CheckError(ierr);
    ierr = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, n, 1, templ,
                              &x_loc);
    CheckError(ierr);
    ierr = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, n, 1, templ,
                              &dist_loc);
    CheckError(ierr);
    ierr = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, m, 1, templ,
                              &yy_loc);
    CheckError(ierr);
    ierr = PLA_Mscalar_create(MPIfloat, PLA_ALL_ROWS, PLA_ALL_COLS, 1, 1, templ,
                              &plus_one);
    CheckError(ierr);
    ierr = PLA_Obj_set_to_one(plus_one);
    CheckError(ierr);

    iter=0;
    // Optimization loop
    while(1)
    {
        // Only one processor does the working set selection.
        int status = 0;
        if(rank == 0)
        {
            if(iter > 0)
            {
                status = select_working_set(work_set, not_work_set);
            }
            else
            {
                init_working_set(work_set, not_work_set);
            }
        }

        // Send status to other processors.
        ierr = MPI_Bcast(&status, 1, MPI_INT, 0, comm);
        // Now check for optimality
        if(status != 0)
            break;
        // Send new opt_precision, as select_working_set might have
        // changed it.
        ierr = MPI_Bcast(&opt_precision, 1, MPI_DOUBLE, 0, comm);
        // Send new working set size and working set to other processors.
        ierr = MPI_Bcast(&n, 1, MPI_INT, 0, comm);
        ierr = MPI_Bcast(work_set, n, MPI_INT, 0, comm);
        lmn = l - n;
        ierr = MPI_Bcast(not_work_set, lmn, MPI_INT, 0, comm);

//       info("working set proc[%d]=", rank);
//       for(int i=0; i<n; ++i)
// 	info(" %d", work_set[i]);
//       info("\n"); info_flush();

//       info("not workset proc[%d]=", rank);
//       for(int i=0; i<lmn; ++i)
// 	info(" %d", not_work_set[i]);
//       info("\n"); info_flush();

        // Recompute ranges, as n and lmn might have changed
        setup_range(n_low, n_up, n);
        this->n_low_loc = n_low[rank];
        this->n_up_loc = n_up[rank];
        setup_range(lmn_low, lmn_up, lmn);
        this->lmn_low_loc = lmn_low[rank];
        this->lmn_up_loc = lmn_up[rank];
        this->local_n = n_up_loc - n_low_loc;
        this->local_lmn = lmn_up_loc - lmn_low_loc;

        // Create new views into existing objs, as n might
        // have changed
        ierr = PLA_Obj_view(Q_bb_global, n, n, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &Q_bb_global_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(c_global, n, 1, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &c_global_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(a_global, m, n, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &a_global_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(up_global, n, 1, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &up_global_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(low_global, n, 1, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &low_global_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(x, n, 1, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &x_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(dist, n, 1, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &dist_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(g, n, 1, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &g_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(t, n, 1, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &t_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(z, n, 1, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &z_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(s, n, 1, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &s_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(x_loc, n, 1, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &x_loc_view);
        CheckError(ierr);
        ierr = PLA_Obj_view(dist_loc, n, 1, PLA_ALIGN_FIRST,
                            PLA_ALIGN_FIRST, &dist_loc_view);
        CheckError(ierr);

        void *addr_buf = NULL;
        ierr = PLA_Obj_local_buffer(x_loc_view, (void **) &addr_buf);
        CheckError(ierr);
        alpha_new = (LOQOfloat *) addr_buf;
        ierr = PLA_Obj_local_buffer(yy_loc, (void **) &addr_buf);
        CheckError(ierr);
        dual = (LOQOfloat *) addr_buf;
        ierr = PLA_Obj_local_buffer(dist_loc_view, (void **) &addr_buf);
        dist_ = (LOQOfloat *) addr_buf;

        ++iter;
        // Setup problem for parallel LOQO solver
        time = MPI_Wtime();
        info("setup_problem...");
        setup_problem(work_set, not_work_set);
        info("done.\n");
        info_flush();
        info("setup_up...");
        setup_up(work_set);
        info("done.\n");
        info_flush();
        info("setup_low...");
        setup_low();
        info("done.\n");
        info_flush();
        //print_problem();
        time = MPI_Wtime() - time;
        problem_setup_time += time;

        // Compute obj before optimization
        info("Computing obj before...");
        info_flush();
        obj_before = 0.0;
        int start_i = 0;
        for(int i=0; i<n; ++i, ++start_i)
        {
            if(work_set[i] >= l_low_loc)
                break;
        }
        //      for(int i=n_low_loc; i<n_up_loc; ++i)
        for(int i=start_i; work_set[i] < l_up_loc && i<n; ++i)
        {
            //obj_before += alpha[work_set[i]]*c[i-n_low_loc];
            obj_before += alpha[work_set[i]]*c[i-start_i];
            for(int j=0; j<n; ++j)
//  	    obj_before += 0.5*alpha[work_set[i]]*Q_bb[local_n*j+(i-n_low_loc)]*
// 	      alpha[work_set[j]];
                obj_before += 0.5*alpha[work_set[i]]*Q_bb[num_rows*j+(i-start_i)]*
                              alpha[work_set[j]];
        }
        // Get local changes for obj_before from other processors
        double obj_before_loc=0.0;
        double obj_before_other=0.0;
        for(int i=0; i<size; ++i)
        {
            if(i == rank)
                ierr = MPI_Bcast(&obj_before, 1, MPI_DOUBLE, i, comm);
            else
            {
                ierr = MPI_Bcast(&obj_before_loc, 1, MPI_DOUBLE, i, comm);
                obj_before_other += obj_before_loc;
            }
        }
        obj_before += obj_before_other;
        printf("obj before = %g\n", obj_before);
        fflush(stdout);
        info("done.\n");
        info_flush();

        // Run solver
        time = MPI_Wtime();
        status = solve_inner(work_set);
        time = MPI_Wtime() - time;
        inner_solver_time += time;

        // Update gradient.
        time = MPI_Wtime();
        int *nz = new int[n];
        for(int i=0; i<n; ++i)
        {
            delta_alpha[i] = alpha_new[i] - alpha[work_set[i]];
            if(fabs(delta_alpha[i]) > TOL_ZERO)
                nz[i] = 1;
            else
                nz[i] = 0;
        }
        info("Updating G_b...");
        info_flush();
        // Compute G_b
        for(int i=start_i; work_set[i] < l_up_loc && i<n; ++i)
            // for(int i=n_low_loc; i<n_up_loc; ++i)
        {
            for(int j=0; j<n; ++j)
            {
                if(nz[j])
// 		G[work_set[i]] += Q_bb[j*local_n+(i-n_low_loc)] *
// 		  delta_alpha[j];
                    G[work_set[i]] += Q_bb[j*num_rows+(i-start_i)] *
                                      delta_alpha[j];
            }
        }
        info("done.\n");
        info_flush();

        // Update local part of the parallel cache status
        for(int i=0; i<l; ++i)
        {
            if(Q.is_cached(i))
                p_cache_status[rank*size + i] = CACHED;
            else
                p_cache_status[rank*size + i] = NOT_CACHED;
        }
        // Synchronize parallel cache status
        for(int k=0; k<size; ++k)
        {
            ierr = MPI_Bcast(&p_cache_status[k*size], l, MPI_CHAR, k, comm);
            CheckError(ierr);
        }
//       printf("p_cache_status %d =",rank);
//       for(int k=0; k<size; ++k)
// 	{
// 	  for(int i=0; i<l; ++i)
// 	    {
// 	      printf(" %d", p_cache_status[k*size+i]);
// 	    }
// 	  printf("\n");
// 	}
//       printf("\n");

        // Smart parallel cache handling
        int *idx_cached = new int[n];
        int *idx_not_cached = new int[n];
        int count_cached = 0;
        int count_not_cached = 0;
        int next_k = 0;
        bool found = false;
        int next_not_cached = 0;
        for(int i=0; i<n; ++i)
        {
            if(nz[i])
            {
                for(int k=next_k; !found && k<size; ++k)
                {
                    if(p_cache_status[k*size + work_set[i]] == CACHED)
                    {
                        if(k == rank) // Do we have it?
                        {
                            idx_cached[count_cached] = i;
                            ++count_cached;
                        }
                        found = true;
                        next_k = k == size-1 ? 0 : k+1;
                    }
                }
                for(int k=0; !found && k<next_k; ++k)
                {
                    if(p_cache_status[k*size + work_set[i]] == CACHED)
                    {
                        if(k == rank) // Do we have it?
                        {
                            idx_cached[count_cached] = i;
                            ++count_cached;
                        }
                        found = true;
                        next_k = k == size-1 ? 0 : k+1;
                    }
                }
                if(!found) // not in any cache
                {
                    if(rank == next_not_cached) // Do we have to compute it?
                    {
                        idx_not_cached[count_not_cached] = i;
                        ++count_not_cached;
                    }
                    next_not_cached = next_not_cached == size-1 ? 0 :
                                      next_not_cached + 1;
                }
                found = false;
            } // if(nz[i])
        }

        info("Updating G_n...");
        info_flush();
        // Compute G_n
        for(int j=0; j<lmn; ++j)
            G_n[j] = 0;
//       printf("idx_cached %d =", rank);
//       for(int i=0; i<count_cached; ++i)
// 	printf(" %d", idx_cached[i]);
//       printf("\n");
//       printf("idx_not_cached %d =", rank);
//       for(int i=0; i<count_not_cached; ++i)
// 	printf(" %d", idx_not_cached[i]);
//       printf("\n");
        // First update the cached part...
        for(int i=0; i<count_cached; ++i)
        {
            const Qfloat *Q_i = Q.get_Q_subset(work_set[idx_cached[i]],
                                               not_work_set,lmn);
            for(int j=0; j<lmn; ++j)
                G_n[j] += Q_i[not_work_set[j]] * delta_alpha[idx_cached[i]];
        }
        // now update the non-cached part
        for(int i=0; i<count_not_cached; ++i)
        {
            const Qfloat *Q_i = Q.get_Q_subset(work_set[idx_not_cached[i]],
                                               not_work_set,lmn);
            for(int j=0; j<lmn; ++j)
                G_n[j] += Q_i[not_work_set[j]] * delta_alpha[idx_not_cached[i]];
        }
//       for(int i=start_i; work_set[i] < l_up_loc && i<n; ++i)
// 	 //      for(int i=n_low_loc; i<n_up_loc; ++i)
// 	{
// 	  if(nz[i]){
// 	    const Qfloat *Q_i = Q.get_Q_subset(work_set[i],not_work_set,lmn);
// 	    for(int j=0; j<lmn; ++j)
// 	      G_n[j] += Q_i[not_work_set[j]] * delta_alpha[i];

// 	  }
// 	}

        info("done.\n");
        info_flush();
        delete[] idx_cached;
        delete[] idx_not_cached;
        delete[] nz;
        time = MPI_Wtime() - time;
        gradient_updating_time += time;

        // Synchronize gradient with other processors
        info("Synchronizing gradient...");
        info_flush();
        sync_gradient(work_set, not_work_set);
        info("done.\n");
        info_flush();

//       info("synced G proc[%d]=",rank);
//       for(int i=0; i<l; ++i)
// 	info(" %g", G[i]);
//       info("\n");
//       info_flush();
        // Update alpha
        for(int i=0; i<n; ++i)
        {
            alpha[work_set[i]] = alpha_new[i];
            update_alpha_status(work_set[i]);
        }
    } // while(1)
    // Calculate rho
    si->rho = calculate_rho();

    // Calculate objective value
    {
        double v = 0;
        int i;
        for(i=0; i<l; i++)
            v += alpha[i] * (G[i] + b[i]);

        si->obj = v/2;
    }

    // Put back the solution
    {
//     printf("alpha = ");
        for(int i=0; i<l; i++)
        {
            alpha_[i] = alpha[i];
// 	printf(" %g\n",alpha[i]);
        }
//     printf("\n");
    }

    si->upper_bound_p = Cp;
    si->upper_bound_n = Cn;

    info("\noptimization finished, #iter = %d\n",iter);

    total_time = MPI_Wtime() - total_time;

    // print timing statistics
    if(rank == 0)
    {
        info("Total opt. time = %lf\n", total_time);
        info_flush();
        info("Problem setup time = %lf (%lf%%)\n", problem_setup_time,
             problem_setup_time/total_time*100);
        info_flush();
        info("Inner solver time = %lf (%lf%%)\n", inner_solver_time,
             inner_solver_time/total_time*100);
        info_flush();
        info("Gradient updating time = %lf (%lf%%)\n", gradient_updating_time,
             gradient_updating_time/total_time*100);
        info_flush();
    }

    // Clean up
    delete[] b;
    delete[] y;
    delete[] alpha;
    delete[] alpha_status;
    delete[] delta_alpha;
    delete[] work_status;
    delete[] work_count;
    delete[] work_set;
    delete[] not_work_set;
    delete[] G;
    delete[] Q_bb;
    delete[] c;
    delete[] up;
    delete[] low;
    delete[] a;
    delete[] d;
    delete[] l_low;
    delete[] l_up;
    delete[] n_low;
    delete[] n_up;
    delete[] lmn_low;
    delete[] lmn_up;
    delete[] G_n;
    delete[] p_cache_status;
    ierr = PLA_Obj_free(&Q_bb_global);
    CheckError(ierr);
    ierr = PLA_Obj_free(&c_global);
    CheckError(ierr);
    ierr = PLA_Obj_free(&up_global);
    CheckError(ierr);
    ierr = PLA_Obj_free(&low_global);
    CheckError(ierr);
    ierr = PLA_Obj_free(&a_global);
    CheckError(ierr);
    ierr = PLA_Obj_free(&d_global);
    CheckError(ierr);
    ierr = PLA_Obj_free(&x_loc);
    CheckError(ierr);
    ierr = PLA_Obj_free(&x);
    CheckError(ierr);
    ierr = PLA_Obj_free(&g);
    CheckError(ierr);
    ierr = PLA_Obj_free(&t);
    CheckError(ierr);
    ierr = PLA_Obj_free(&yy);
    CheckError(ierr)
    ierr = PLA_Obj_free(&yy_loc);
    CheckError(ierr);
    ierr = PLA_Obj_free(&z);
    CheckError(ierr);
    ierr = PLA_Obj_free(&s);
    CheckError(ierr);
}

void Solver_Parallel_LOQO::sync_gradient(int *work_set, int *not_work_set)
{
    int start_i = 0;
    for(int i=0; i<n; ++i, ++start_i)
    {
        if(work_set[i] >= l_low_loc)
            break;
    }
    // Synchronize G_b
    //  double *G_send = new double[local_n];
    double *G_send = new double[num_rows];
    double *G_recv = new double[n];
    int count=0;
    for(int i=start_i; work_set[i] < l_up_loc && i<n; ++i)
        //for(int i=n_low_loc; i<n_up_loc; ++i)
    {
        G_send[count] = G[work_set[i]];
        ++count;
    }
//   printf("G_send = ");
//   for(int i=start_i; work_set[i] < l_up_loc && i<n; ++i)
//     printf(" %g", G_send[i]);
//   printf("\n");
//   printf("count = %d\n", count);
//   printf("num_rows = %d\n", num_rows);
    int *sendcounts = new int[size];
    int *sdispls = new int[size];
    for(int i=0; i<size; ++i)
    {
        //      sendcounts[i] = local_n;
        sendcounts[i] = num_rows;
        sdispls[i] = 0;
    }
    int *recvcounts = new int[size];
    int *rdispls = new int[size];

    for(int k=0; k<size; ++k)
    {
        int start_i_other = 0;
        for(int i=0; i<n; ++i, ++start_i_other)
            if(work_set[i] >= l_low[k])
                break;
        int num_rows_other = 0;
        for(int i=start_i_other; work_set[i] < l_up[k] && i<n; ++i)
            ++num_rows_other;
        recvcounts[k] = num_rows_other;
        if(k == 0)
            rdispls[k] = 0;
        else
            rdispls[k] = rdispls[k-1] + recvcounts[k-1];
    }
//   recvcounts[0] = n_up[0] - n_low[0];
//   rdispls[0] = 0;
//   for(int i=1; i<size; ++i)
//     {
//       recvcounts[i] = n_up[i] - n_low[i];
//       rdispls[i] = rdispls[i-1] + recvcounts[i-1];
//     }
    ierr = MPI_Alltoallv(G_send, sendcounts, sdispls, MPI_DOUBLE,
                         G_recv, recvcounts, rdispls, MPI_DOUBLE, comm);
    CheckError(ierr);
    // Update local G_b
    for(int k=0; k<size; ++k)
    {
        if(k != rank)
        {
            int start_i_other = 0;
            for(int i=0; i<n; ++i, ++start_i_other)
                if(work_set[i] >= l_low[k])
                    break;
            // G_b
            for(int i=start_i_other; work_set[i] < l_up[k] && i<n; ++i)
                G[work_set[i]] = G_recv[rdispls[k]+(i-start_i_other)];
// 	  for(int j=n_low[i]; j<n_up[i]; ++j)
// 	    G[work_set[j]] = G_recv[rdispls[i]+(j-n_low[i])];
        }
    }
    delete[] G_send;
    delete[] G_recv;
    delete[] sendcounts;
    delete[] sdispls;
    delete[] recvcounts;
    delete[] rdispls;

    double *G_buf = new double[lmn];
    // Synchronize G_n
    for(int i=0; i<size; ++i)
    {
        if(rank == i)
        {
            for(int j=0; j<lmn; ++j)
                G_buf[j] = G_n[j];
        }
        ierr = MPI_Bcast(G_buf, lmn, MPI_DOUBLE, i, comm);
        CheckError(ierr);
        // Accumulate contributions
        for(int j=0; j<lmn; ++j)
            G[not_work_set[j]] += G_buf[j];
    }
    delete[] G_buf;
}

int Solver_LOQO::select_working_set(int *work_set, int *not_work_set)
{
    printf("selecting working set...");
    // reset work status
    n = n_old;
    for(int i=0; i<l; ++i)
        work_status[i] = WORK_N;

    double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
    double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }
    for(int i=0; i<l; ++i)
    {
        if(!is_upper_bound(i))
        {
            if(y[i] == +1)
            {
                if(-G[i] > Gmax1)
                    Gmax1 = -G[i];
            }
            else
            {
                if(-G[i] > Gmax2)
                    Gmax2 = -G[i];
            }
        }
        if(!is_lower_bound(i))
        {
            if(y[i] == +1)
            {
                if(G[i] > Gmax2)
                    Gmax2 = G[i];
            }
            else
            {
                if(G[i] > Gmax1)
                    Gmax1 = G[i];
            }
        }
    }
    // check for optimality
    printf("Gmax1 + Gmax2 = %g < %g\n", Gmax1+Gmax2,eps);
    if(Gmax1 + Gmax2 < eps)
        return 1;

    // Compute yG
    double *yG = new double[l];
    int *pidx = new int[l];
    for(int i=0; i<l; ++i)
    {
        if(y[i] == +1)
            yG[i] = G[i];
        else
            yG[i] = -G[i];
        pidx[i] = i;
    }
    quick_sort(yG, pidx, 0, l-1);
//   printf("yG = ");
//   for(int i=0; i<l; ++i)
//     printf(" %g",yG[i]);
//   printf("\n");
//   printf("pidx = ");
//   for(int i=0; i<l; ++i)
//     printf(" %d",pidx[i]);
//   printf("\n");
    int top=l-1;
    int bot=0;
    int count=0;
    // Select a full set initially
    int nselect = iter == 0 ? n : q;
    while(top > bot && count < nselect)
    {
        while(!(is_free(pidx[top])
                || (is_upper_bound(pidx[top]) && y[pidx[top]] == +1)
                || (is_lower_bound(pidx[top]) && y[pidx[top]] == -1)
               ))
            --top;
        while(!(is_free(pidx[bot])
                || (is_upper_bound(pidx[bot]) && y[pidx[bot]] == -1)
                || (is_lower_bound(pidx[bot]) && y[pidx[bot]] == +1)
               ))
            ++bot;
        if(top > bot)
        {
            count += 2;
            work_status[pidx[top]] = WORK_B;
            work_status[pidx[bot]] = WORK_B;
            --top;
            ++bot;
        }
    }
    if(count < n)
    {
        // Compute subset of indices in previous working set
        // which were not yet selected
        int j=0;
        int *work_count_subset = new int[l-count];
        int *subset = new int[l-count];
        int *psubset = new int[l-count];
        for(int i=0; i<l; ++i)
        {
            if(work_status[i] == WORK_N && work_count[i] > -1)
            {
                work_count_subset[j] = work_count[i];
                subset[j] = i;
                psubset[j] = j;
                ++j;
            }
        }
        quick_sort(work_count_subset, psubset, 0, j-1);

        // Fill up with j \in B, 0 < alpha[j] < C
        for(int i=0; i<j; ++i)
        {
            if(count == n) break;
            if(is_free(subset[psubset[i]]))
            {
                work_status[subset[psubset[i]]] = WORK_B;
                ++count;
            }
        }
        // Fill up with j \in B, alpha[j] = 0
        for(int i=0; i<j; ++i)
        {
            if(count == n) break;
            if(is_lower_bound(subset[psubset[i]]))
            {
                work_status[subset[psubset[i]]] = WORK_B;
                ++count;
            }
        }
        // Fill up with j \in B, alpha[j] = C
        for(int i=0; i<j; ++i)
        {
            if(count == n) break;
            if(is_upper_bound(subset[psubset[i]]))
            {
                work_status[subset[psubset[i]]] = WORK_B;
                ++count;
            }
        }
        // clean up
        delete[] work_count_subset;
        delete[] subset;
        delete[] psubset;
    } // if(count < n)
    // Setup work_set and not_work_set
    // update work_count
    int nnew=0;
    int i=0;
    int j=0;
    n=0;
    for(int t=0; t<l; ++t)
    {
        if(work_status[t] == WORK_B)
        {
            if(work_count[t] == -1)
                ++nnew;
            work_set[i] = t;
            ++i;
            ++n;
            ++work_count[t];
        }
        else
        {
            not_work_set[j] = t;
            ++j;
            work_count[t] = -1;
        }
    }
    // Update q
    printf("nnew = %d\n", nnew);
    int kin = nnew;
    nnew = nnew % 2 == 0 ? nnew : nnew-1;
    int L = n/10 % 2 == 0 ? n/10 : (n/10)-1;
    q = min(q, max( max( 10, L ), nnew ) );
    printf("q = %d\n", q);
    printf("n = %d\n",n);
    if(kin == 0)
    {
        // 1st: Increase precision of solver.
        if(opt_precision > 1e-20)
            opt_precision /= 100;
        else
        {
            info("Error: Unable to select a suitable working set!!!\n");
            return 1;
        }
    }
    // Clean up
    delete[] yG;
    delete[] pidx;
    printf("done.\n");
    return 0;
}

class Solver_LOQO_NU : public Solver_LOQO
{
public:
    // Ensure minimum working set size =3,
    // next even =4.
    Solver_LOQO_NU(int n, int q, int m, double nu) : Solver_LOQO(max(4,n),q, m)
    {
        this->nu = nu;
    };
    void Solve(int l, const QMatrix& Q, const double *b, const schar *y,
               double *alpha, double Cp, double Cn, double eps,
               SolutionInfo* si, int shrinking)
    {
        Solver_LOQO::Solve(l,Q,b,y,alpha,Cp,Cn,eps,si,shrinking);
    }
private:
    double nu;
    void allocate_a() {
        a = new double[2*n];
    }
    void allocate_d() {
        d = new double[2];
    }
    void allocate_work_space() {
        work_space = new double[5*n+2];
    }
    void setup_a(int *work_set);
    void setup_d(int *not_work_set);
    double calculate_rho();
    //double calculate_rho(){ si->r = work_space[3*n+1]; return work_space[3*n]; }
    int check_working_set();
    int select_working_set(int *work_set, int *not_work_set);
    void init_working_set(int *work_set, int *not_work_set);
};

void Solver_LOQO_NU::init_working_set(int *work_set, int *not_work_set)
{
    int j;
    NEXT_RAND = 1;
    for (int i=0; i<n; ++i)
    {
        do
        {
            j = next_rand_pos() % l;
        } while (work_status[j] != WORK_N);
        work_status[j] = WORK_B;
    }
    int k=0;
    j=0;
    for(int i=0; i<l; ++i)
    {
        if(work_status[i] == WORK_B)
        {
            work_set[j] = i;
            ++j;
            work_count[i] = 0;
        }
        else
        {
            not_work_set[k] = i;
            ++k;
            work_count[i] = -1;
        }
    }
}

void Solver_LOQO_NU::setup_a(int *work_set)
{
    for(int i=0; i<n; ++i)
    {
        a[i] = y[work_set[i]];
        a[n+i] = 1;
    }
}

void Solver_LOQO_NU::setup_d(int *not_work_set)
{
    d[0]=0;
    for(int i=0; i<lmn; ++i)
        d[0] -= y[not_work_set[i]]*alpha[not_work_set[i]];
    d[1] = nu*l;
    for(int i=0; i<lmn; ++i)
        d[1] -= alpha[not_work_set[i]];
}

int Solver_LOQO_NU::select_working_set(int *work_set, int *not_work_set) {
    printf("selecting working set...");
    // reset work status
    n = n_old;
    for(int i=0; i<l; ++i)
    {
        work_status[i] = WORK_N;
    }
    double Gmin1 = INF;
    double Gmin2 = INF;
    double Gmax1 = -INF;
    double Gmax2 = -INF;
    int min1 = -1;
    int min2 = -1;
    int max1 = -1;
    int max2 = -1;
    for(int t=0; t<l; ++t)
    {
        if(y[t] == +1)
        {
            if(!is_upper_bound(t))
            {
                if(G[t] < Gmin1)
                {
                    Gmin1 = G[t];
                    min1 = t;
                }
            }
            if(!is_lower_bound(t))
            {
                if(G[t] > Gmax1)
                {
                    Gmax1 = G[t];
                    max1 = t;
                }
            }
        }
        else
        {
            if(!is_upper_bound(t))
            {
                if(G[t] < Gmin2)
                {
                    Gmin2 = G[t];
                    min2 = t;
                }
            }
            if(!is_lower_bound(t))
            {
                if(G[t] > Gmax2)
                {
                    Gmax2 = G[t];
                    max2 = t;
                }
            }
        }
    }
    // check for optimality, max. violating pair.
    printf("max(Gmax1-Gmin1,Gmax2-Gmin2) = %g < %g\n",
           max(Gmax1-Gmin1,Gmax2-Gmin2),eps);
    if(max(Gmax1-Gmin1,Gmax2-Gmin2) < eps)
        return 1;

    // Sort gradient
    double *Gtmp = new double[l];
    int *pidx = new int[l];
    for(int i=0; i<l; ++i)
    {
        Gtmp[i] = G[i];
        pidx[i] = i;
    }
    quick_sort(Gtmp, pidx, 0, l-1);
//   printf("pidx = ");
//   for(int i=0; i<l; ++i)
//     printf(" %d", pidx[i]);
//   printf("\n");

    int top1=l-1;
    int top2=l-1;
    int bot1=0;
    int bot2=0;
    int count=0;
    // Select a full set initially
    int nselect = iter == 0 ? n : q;
    while(count < nselect)
    {
        if(top1 > bot1)
        {
            while(!( (is_free(pidx[top1]) || is_upper_bound(pidx[top1]))
                     && y[pidx[top1]] == +1))
            {
                if(top1 <= bot1) break;
                --top1;
            }
            while(!( (is_free(pidx[bot1]) || is_lower_bound(pidx[bot1]))
                     && y[pidx[bot1]] == +1))
            {
                if(bot1 >= top1) break;
                ++bot1;
            }
        }
        if(top2 > bot2)
        {
            while(!( (is_free(pidx[top2]) || is_upper_bound(pidx[top2]))
                     && y[pidx[top2]] == -1))
            {
                if(top2 <= bot2) break;
                --top2;
            }
            while(!( (is_free(pidx[bot2]) || is_lower_bound(pidx[bot2]))
                     && y[pidx[bot2]] == -1))
            {
                if(bot2 >= top2) break;
                ++bot2;
            }
        }
        if(top1 > bot1 && top2 > bot2)
        {
            if(G[pidx[top1]]-G[pidx[bot1]] > G[pidx[top2]]-G[pidx[bot2]])
            {
                work_status[pidx[top1]] = WORK_B;
                work_status[pidx[bot1]] = WORK_B;
                --top1;
                ++bot1;
            }
            else
            {
                work_status[pidx[top2]] = WORK_B;
                work_status[pidx[bot2]] = WORK_B;
                --top2;
                ++bot2;
            }
            count += 2;
        }
        else if(top1 > bot1)
        {
            work_status[pidx[top1]] = WORK_B;
            work_status[pidx[bot1]] = WORK_B;
            --top1;
            ++bot1;
            count += 2;
        }
        else if(top2 > bot2)
        {
            work_status[pidx[top2]] = WORK_B;
            work_status[pidx[bot2]] = WORK_B;
            --top2;
            ++bot2;
            count += 2;
        }
        else
            break;
    } // while(count < nselect)
    if(count < n)
    {
        // Compute subset of indices in previous working set
        // which were not yet selected
        int j=0;
        int *work_count_subset = new int[l-count];
        int *subset = new int[l-count];
        int *psubset = new int[l-count];
        for(int i=0; i<l; ++i)
        {
            if(work_status[i] == WORK_N && work_count[i] > -1)
            {
                work_count_subset[j] = work_count[i];
                subset[j] = i;
                psubset[j] = j;
                ++j;
            }
        }
        quick_sort(work_count_subset, psubset, 0, j-1);

        // Fill up with j \in B, 0 < alpha[j] < C
        for(int i=0; i<j; ++i)
        {
            if(count == n) break;
            if(is_free(subset[psubset[i]]))
            {
                work_status[subset[psubset[i]]] = WORK_B;
                ++count;
            }
        }
        // Fill up with j \in B, alpha[j] = 0
        for(int i=0; i<j; ++i)
        {
            if(count == n) break;
            if(is_lower_bound(subset[psubset[i]]))
            {
                work_status[subset[psubset[i]]] = WORK_B;
                ++count;
            }
        }
        // Fill up with j \in B, alpha[j] = C
        for(int i=0; i<j; ++i)
        {
            if(count == n) break;
            if(is_upper_bound(subset[psubset[i]]))
            {
                work_status[subset[psubset[i]]] = WORK_B;
                ++count;
            }
        }
        // clean up
        delete[] work_count_subset;
        delete[] subset;
        delete[] psubset;
    } // if(count < n)
    // Setup work_set and not_work_set
    // update work_count
    int nnew=0;
    int i=0;
    int j=0;
    n=0;
    for(int t=0; t<l; ++t)
    {
        if(work_status[t] == WORK_B)
        {
            if(work_count[t] == -1)
                ++nnew;
            work_set[i] = t;
            ++i;
            ++n;
            ++work_count[t];
        }
        else
        {
            not_work_set[j] = t;
            ++j;
            work_count[t] = -1;
        }
    }
    // Update q
    printf("nnew = %d\n", nnew);
    int kin = nnew;
    nnew = nnew % 2 == 0 ? nnew : nnew-1;
    int L = n/10 % 2 == 0 ? n/10 : (n/10)-1;
    q = min(q, max( max( 10, L ), nnew ) );
    printf("q = %d\n", q);
    printf("n = %d\n",n);
    if(kin == 0)
    {
        // 1st: Increase precision of solver.
        if(opt_precision > 1e-20)
            opt_precision /= 100;
        else
        {
            info("Error: Unable to select a suitable working set!!!\n");
            return 1;
        }
    }
    // clean up
    delete[] Gtmp;
    delete[] pidx;
    printf("done.\n");
    return 0;
}

void Solver_LOQO::setup_a(int *work_set)
{
    info("Solver_LOQO::setting up a...");
    info_flush();
    for(int i=0; i<n; ++i)
    {
        a[i] = y[work_set[i]];
    }
    info("done.\n");
    info_flush();
}

void Solver_LOQO::setup_d(int *not_work_set)
{
    info("setting up d...");
    info_flush();
    d[0] = 0;
    for(int i=0; i<lmn; ++i)
    {
        if(fabs(alpha[not_work_set[i]]) > TOL_ZERO)
            d[0] -= y[not_work_set[i]]*alpha[not_work_set[i]];
    }
    info("done.\n");
    info_flush();
}

void Solver_LOQO::setup_problem(int *work_set, int *not_work_set)
{
    info("setting up Q_bb...");
    info_flush();
    for(int i=0; i<n; ++i)
    {
        const Qfloat *Q_i = Q->get_Q_subset(work_set[i],work_set,n);
        //  	  const Qfloat *Q_i = Q.get_Q(work_set[i],l);
        c[i] = G[work_set[i]];
        for(int j=0; j<n; ++j)
        {
            Q_bb[i*n+j] = (double) Q_i[work_set[j]];
            //   	      Q_bb[i*n+j] = Q_i[work_set[j]];
            if(alpha[work_set[j]] > TOL_ZERO)
                c[i] -= Q_bb[i*n+j]*alpha[work_set[j]];
        }
    }
    setup_a(work_set);
    setup_d(not_work_set);
    info("done.\n");
    info_flush();
}

void Solver_LOQO::print_problem()
{
    printf("G=");
    for(int i=0; i<l; ++i)
    {
        printf(" %g", G[i]);
    }
    printf("\n");
//   printf("Q_bb=");
//   for(int i=0; i<n; ++i)
//     {
//       for(int j=0; j<n; ++j)
// 	printf(" %g", Q_bb[i*n+j]);
//       printf("\n");
//     }
    printf("d[0] = % g\n",d[0]);
    printf("d[1] = % g\n",d[1]);
    printf("a=");
    for(int i=0; i<n; ++i)
        printf(" %g", a[i]);
    printf("\n");

    printf("a2=");
    for(int i=0; i<n; ++i)
        printf(" %g", a[i+n]);
    printf("\n");

    printf("c=");
    for(int i=0; i<n; ++i)
        printf(" %g", c[i]);
    printf("\n");
    printf("low=");
    for(int i=0; i<n; ++i)
        printf(" %g", low[i]);
    printf("\n");
    printf("up=");
    for(int i=0; i<n; ++i)
        printf(" %g", up[i]);
    printf("\n");
}

double Solver_LOQO_NU::calculate_rho()
{
    int nr_free1 = 0,nr_free2 = 0;
    double ub1 = INF, ub2 = INF;
    double lb1 = -INF, lb2 = -INF;
    double sum_free1 = 0, sum_free2 = 0;

    for(int i=0; i<active_size; i++)
    {
        if(y[i]==+1)
        {
            if(is_lower_bound(i))
                ub1 = min(ub1,G[i]);
            else if(is_upper_bound(i))
                lb1 = max(lb1,G[i]);
            else
            {
                ++nr_free1;
                sum_free1 += G[i];
            }
        }
        else
        {
            if(is_lower_bound(i))
                ub2 = min(ub2,G[i]);
            else if(is_upper_bound(i))
                lb2 = max(lb2,G[i]);
            else
            {
                ++nr_free2;
                sum_free2 += G[i];
            }
        }
    }
    printf("nr_free1 = %d\n", nr_free1);
    printf("sum_free1 = %g\n",sum_free1);
    printf("nr_free2 = %d\n", nr_free2);
    printf("sum_freee = %g\n",sum_free2);
    double r1,r2;
    if(nr_free1 > 0)
        r1 = sum_free1/nr_free1;
    else
        r1 = (ub1+lb1)/2;

    if(nr_free2 > 0)
        r2 = sum_free2/nr_free2;
    else
        r2 = (ub2+lb2)/2;

    si->r = (r1+r2)/2;
    printf("(r1+r2)/2 = %g\n", (r1+r2)/2);
    printf("(r1+r2)/2 = %g\n", (r1-r2)/2);
    return (r1-r2)/2;
}

void Solver_LOQO::init_working_set(int *work_set, int *not_work_set)
{
    int j;
    NEXT_RAND = 1;
    for (int i=0; i<n; ++i)
    {
        do
        {
            j = next_rand_pos() % l;
        } while (work_status[j] != WORK_N);
        work_status[j] = WORK_B;
    }
    int k=0;
    j=0;
    for(int i=0; i<l; ++i)
    {
        if(work_status[i] == WORK_B)
        {
            work_set[j] = i;
            ++j;
            work_count[i] = 0;
        }
        else
        {
            not_work_set[k] = i;
            ++k;
            work_count[i] = -1;
        }
    }
}

int Solver_LOQO::solve_inner()
{
    Solver_NU sl;
    schar *sy = new schar[n];
    Qfloat *Q_bb_f = new Qfloat[n*n];
    Qfloat *QD_f = new Qfloat[n];
    for(int i=0; i<n; ++i)
    {
        sy[i] = (schar) a[i];
        QD_f[i] = (Qfloat) Q_bb[i*n+i];
        for(int j=0; j<n; ++j)
            Q_bb_f[i*n+j] = (Qfloat) Q_bb[i*n+j];
    }

    sl.Solve(n, SVQ_No_Cache(Q_bb_f, QD_f, n), c, sy, work_space,
             Cp, Cn, eps, si, /* shrinking */ 0);
    delete[] sy;
    delete[] QD_f;
    return 0;
}

// int Solver_LOQO::solve_inner()
// {
//   int result = 0;
//   double sigdig;
//   double epsilon_loqo = 1e-10;
//   double margin;
//   int iteration;
//   for(margin=init_margin, iteration=init_iter;
//       margin <= 0.9999999 && result != OPTIMAL_SOLUTION;)
//     {
//       sigdig = -log10(opt_precision);
//       // run pr_loqo
//       result = pr_loqo(n, m, c, Q_bb, a, d, low, up, work_space,
// 		       &work_space[3*n], dist, 3, sigdig, iteration, margin,
// 		       up[0]/4, 0);
//       // Note: No need to check for choldc problem, as we employ
//       // Manteuffel shifting.
//       if(result == -1)
// 	{
// 	  info("NOTICE: Restarting PR_LOQO with more conservative");
// 	  info("parameters.\n"); info_flush();
// 	  if(init_margin < 0.80)
// 	    init_margin = (4.0*margin+1.0)/5.0;
// 	  margin = (margin+1.0)/2.0;
// 	  opt_precision *= 10.0;
// 	  info("NOTICE: Reducing precision of PR_LOQO.\n");
// 	}
//       else if(result != OPTIMAL_SOLUTION)
// 	{
// 	  // increase number of iterations
// 	  iteration += 2000;
// 	  init_iter += 10;
// 	  // reduce precision
// 	  opt_precision *= 10.0;
// 	  info("NOTICE: Reducing precision of PR_LOQO!\n"); info_flush();
// 	}
//     }
//   // Check precision of alphas
//   for(int i=0; i<n; ++i)
//     {
//       if(work_space[i] < up[i]-epsilon_loqo && dist[i] < -eps)
// 	epsilon_loqo = 2*(up[i]-work_space[i]);
//       else if(work_space[i] > low[i]+epsilon_loqo && dist[i] > eps)
// 	epsilon_loqo = 2*work_space[i];
//     }
//   info("Using epsilon_loqo= %g\n", epsilon_loqo); info_flush();
//   // Clip alphas to bounds
//   for(int i=0; i<n; ++i)
//     {
//       if(fabs(work_space[i]) < epsilon_loqo)
// 	{
// 	  work_space[i] = 0;
// 	}
//       if(fabs(work_space[i]-get_C(i)) < epsilon_loqo)
// 	{
// 	  work_space[i] = get_C(i);
// 	}
//     }
//   // Compute obj after optimization
//   double obj_after = 0.0;
//   for(int i=0; i<n; ++i)
//     {
//       obj_after += work_space[i]*c[i];
//       obj_after += 0.5*work_space[i]*work_space[i]*Q_bb[i*n+i];
//       for(int j=0; j<i; ++j)
// 	{
// 	  obj_after += work_space[j]*work_space[i]*Q_bb[j*n+i];
// 	}
//     }
//   printf("obj_after/before = %g / %g\n", obj_after, obj_before); fflush(stdout);
//   printf("delta a/b = %g\n", obj_after-obj_before); fflush(stdout);
//   // Check for progress
//   if(obj_after >= obj_before)
//     {
//       // Increase precision
//       opt_precision /= 100.0;
//       ++precision_violations;
//       info("NOTICE: Increasing Precision of PR_LOQO.\n"); info_flush();
//     }
//   if(precision_violations > 500)
//     {
//       // Relax stopping criterion
//       eps *= 10.0;
//       precision_violations = 0;
//       info("WARNING: Relaxing epsilon on KKT-Conditions.\n"); info_flush();
//     }
//   return result;
// }

void Solver_LOQO::Solve(int l, const QMatrix& Q, const double *b_,
                        const schar *y_, double *alpha_, double Cp,
                        double Cn, double eps, SolutionInfo* si,
                        int shrinking)
{
    this->l = l;
    this->si = si;
    this->Q = &Q;
    QD = Q.get_QD(); // we need diagonal for working_set_selection
    clone(b,b_,l);
    clone(y,y_,l);
    clone(alpha,alpha_,l);
    this->Cp = Cp;
    this->Cn = Cn;
    this->eps = eps;
    this->active_size = l;
    this->lmn = l - n;
    int *work_set = new int[n];
    int *not_work_set = new int[l];
    double *delta_alpha = new double[n];
    // init alpha status and work status
    {
        alpha_status = new char[l];
        work_status = new char[l];
        work_count = new int[l];
        for(int i=0; i<l; ++i)
        {
            update_alpha_status(i);
            work_status[i] = WORK_N;
            work_count[i] = -1;
        }
    }
    // init gradient
    info("initializing gradient...");
    info_flush();
    {
        G = new double[l];
        for(int i=0; i<l; ++i)
        {
            G[i] = b[i];
        }
        for(int i=0; i<l; ++i)
        {
            if(!is_lower_bound(i))
            {
                const Qfloat *Q_i = Q.get_Q(i,l);
                double alpha_i = alpha[i];
                for(int j=0; j<l; ++j)
                {
                    G[j] += alpha_i * Q_i[j];
                }
            }
        }
    }
    info("done.\n");
    info_flush();

    // Allocate space for pr_loqo
    Q_bb = new LOQOfloat[n*n];
    c = new LOQOfloat[n];
    up = new LOQOfloat[n];
    low = new LOQOfloat[n];
    dist = new LOQOfloat[n];
    allocate_a();
    allocate_d();
    allocate_work_space();
    iter=0;
    // optimization loop
    while(1)
    {
        // select working set and check for optimality
        if(iter > 0)
        {
            if(select_working_set(work_set, not_work_set) != 0)
                break;
        }
        else
        {
            info("initializing working set...");
            info_flush();
            init_working_set(work_set, not_work_set);
            info("done.\n");
            info_flush();
        }

        lmn = l - n;
//       printf("working_set=");
//       for(int i=0; i<n; ++i)
// 	{
// 	  printf("%d ",work_set[i]);
// 	}
//       printf("\n");
//       printf("not_working_set=");
//       for(int i=0; i<lmn; ++i)
// 	{
// 	  printf("%d ",not_work_set[i]);
// 	}
//       printf("\n");

        ++iter;
        // setup problem for pr_loqo
        info("setting up problem for pr_loqo...");
        info_flush();
        setup_problem(work_set, not_work_set);
        setup_up(work_set);
        setup_low();
//       print_problem();
        info("done.\n");
        // run inner solver
        for(int i=0; i<n; ++i)
            work_space[i]=alpha[work_set[i]];

        // Compute obj before optimization
        obj_before = 0.0;
        for(int i=0; i<n; ++i)
        {
            obj_before += alpha[work_set[i]]*c[i];
            obj_before += 0.5*alpha[work_set[i]]*alpha[work_set[i]]*Q_bb[i*n+i];
            for(int j=0; j<i; ++j)
            {
                obj_before += alpha[work_set[j]]*alpha[work_set[i]]*Q_bb[j*n+i];
            }
        }
        printf("obj_before = %g\n", obj_before);
        fflush(stdout);

        int status = solve_inner();
        printf("pr_loqo status = %d\n",status);

        // Restore Q_bb, lower triangle, overwritten by pr_loqo
        for(int i=0; i<n; ++i)
        {
            for(int j=i+1; j<n; ++j)
            {
                Q_bb[n*j+i] = Q_bb[n*i+j];
            }
        }
        // update gradient, compute G_b += Q_bb*delta_alpha and
        // G_n += Q_nb*delta_alpha.
        int *nz = new int[n];
        double sum_delta_alpha_pos=0;
        double sum_delta_alpha_neg=0;
        for(int i=0; i<n; ++i)
        {
            delta_alpha[i] = work_space[i] - alpha[work_set[i]];
            if(y[work_set[i]]==+1)
                sum_delta_alpha_pos += delta_alpha[i];
            else
                sum_delta_alpha_neg += delta_alpha[i];
            if(fabs(delta_alpha[i]) > TOL_ZERO)
                nz[i] = 1;
            else
                nz[i] = 0;
        }
        printf("sum_delta_alpha_pos = %g\n", sum_delta_alpha_pos);
        printf("sum_delta_alpha_neg = %g\n", sum_delta_alpha_neg);
        for(int i=0; i<n; ++i)
            for(int j=0; j<n; ++j) // G_b
            {
                if(nz[j])
                    G[work_set[i]] += Q_bb[n*i+j]*delta_alpha[j];
            }

        for(int i=0; i<n; ++i) // G_n
        {
            if(nz[i])
            {
                const Qfloat *Q_i =
                    Q.get_Q_subset(work_set[i],not_work_set,lmn);
//  	      const Qfloat *Q_i = Q.get_Q(work_set[i],l);
                for(int j=0; j<lmn; ++j)
                    G[not_work_set[j]] += Q_i[not_work_set[j]] * delta_alpha[i];
            }
        }
        delete[] nz;
        // update alpha
        for(int i=0; i<n; ++i)
        {
            alpha[work_set[i]] = work_space[i];
            update_alpha_status(work_set[i]);
        }
//       double sum_alpha=0;
//       printf("alpha = ");
//       double sum_alpha_pos=0;
//       double sum_alpha_neg=0;
//       for(int i=0; i<l; ++i)
// 	{
// 	  sum_alpha += alpha[i];
// 	  if(y[i]==+1)
// 	    sum_alpha_pos += alpha[i];
// 	  else
// 	    sum_alpha_neg += alpha[i];
// 	  printf(" %g", alpha[i]);
// 	}
//       printf("\n");
//       printf("sum_alpha = %g\n",sum_alpha);
//       printf("sum_alpha_pos = %g\n",sum_alpha_pos);
//       printf("sum_alpha_neg = %g\n",sum_alpha_neg);
//       double delta_alpha_pos = (sum_alpha/2)-sum_alpha_pos;
//       double delta_alpha_neg = (sum_alpha/2)-sum_alpha_neg;
        // Restore consistency
//       if(fabs(delta_alpha_pos) > 0)
// 	{
// 	  for(int i=0; i<l; ++i)
// 	    {
// 	      if(y[i] == +1)
// 		{
// 		  if(alpha[i]+delta_alpha_pos >= low[i] &&
// 		     alpha[i]+delta_alpha_pos <= up[i])
// 		    {
// 		      alpha[i] += delta_alpha_pos;
// 		      break;
// 		    }
// 		}
// 	    }
// 	}
//       if(fabs(delta_alpha_neg) > 0)
// 	{
// 	  for(int i=0; i<l; ++i)
// 	    {
// 	      if(y[i] == -1)
// 		{
// 		  if(alpha[i]+delta_alpha_neg >= low[i] &&
// 		     alpha[i]+delta_alpha_neg <= up[i])
// 		    {
// 		      alpha[i] += delta_alpha_neg;
// 		      break;
// 		    }
// 		}
// 	    }
// 	}
        // Recheck
//       sum_alpha_pos = 0;
//       sum_alpha_neg = 0;
//       sum_alpha = 0;
//       for(int i=0; i<l; ++i)
// 	{
// 	  sum_alpha += alpha[i];
// 	  if(y[i] == +1)
// 	    sum_alpha_pos += alpha[i];
// 	  else
// 	    sum_alpha_neg += alpha[i];
// 	}
//       printf("sum_alpha = %g\n",sum_alpha);
//       printf("sum_alpha_pos = %g\n",sum_alpha_pos);
//       printf("sum_alpha_neg = %g\n",sum_alpha_neg);
    } // while(1)

    // calculate rho
    si->rho=calculate_rho();

    // calculate objective value
    {
        double v = 0;
        int i;
        for(i=0; i<l; i++)
            v += alpha[i] * (G[i] + b[i]);

        si->obj = v/2;
    }

    // put back the solution
    {
        for(int i=0; i<l; i++)
            alpha_[i] = alpha[i];
    }

    si->upper_bound_p = Cp;
    si->upper_bound_n = Cn;

    info("\noptimization finished, #iter = %d\n",iter);

    // clean up
    delete[] b;
    delete[] y;
    delete[] alpha;
    delete[] alpha_status;
    delete[] delta_alpha;
    delete[] work_status;
    delete[] work_count;
    delete[] work_set;
    delete[] not_work_set;
    delete[] G;
    delete[] Q_bb;
    delete[] c;
    delete[] up;
    delete[] low;
    delete[] dist;
    delete[] a;
    delete[] d;
    delete[] work_space;
}
