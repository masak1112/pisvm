// D. Brugger, december 2006
// $Id: psmo_solver.cpp 573 2010-12-29 10:54:20Z dome $
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

#include "psmo_solver.h"

#define CheckError(n) if(n){printf("line %d, file %s\n",__LINE__,__FILE__);}
#define MPIfloat MPI_DOUBLE

const double Solver_Parallel_SMO::TOL_ZERO = 1e-09; // tolerance for zero entries

Solver_Parallel_SMO::Solver_Parallel_SMO(int n, int q, MPI_Comm comm)
{
    // Ensure that n,q are even numbers.
    this->n = n % 2 == 0 ? n : n-1;
    this->n_old = this->n;
    this->q = q % 2 == 0 ? q : q-1;
    // Ensure sane q
    this->q = this->q > this->n ? this->n : this->q;
    this->comm = comm;
    ierr = MPI_Comm_rank(comm, &this->rank);
    CheckError(ierr);
    ierr = MPI_Comm_size(comm, &this->size);
    CheckError(ierr);
    NEXT_RAND = 1;
}

unsigned int Solver_Parallel_SMO::next_rand_pos()
{
    NEXT_RAND = NEXT_RAND*1103515245L + 12345L;
    return NEXT_RAND & 0x7fffffff;
}

Solver_Parallel_SMO_NU::Solver_Parallel_SMO_NU(int n, int q, MPI_Comm comm)
    : Solver_Parallel_SMO(n,q,comm)
{}

double Solver_Parallel_SMO_NU::calculate_rho()
{
    int nr_free1 = 0,nr_free2 = 0;
    double ub1 = INF, ub2 = INF;
    double lb1 = -INF, lb2 = -INF;
    double sum_free1 = 0, sum_free2 = 0;

//   printf("alpha = ");
//   for(int i=0; i<l; ++i)
//     printf(" %g",alpha[i]);
//   printf("\n");

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

void Solver_Parallel_SMO_NU::solve_inner()
{
    Solver_NU sl;
    sl.Solve(n, SVQ_No_Cache(Q_bb, QD_b, n), c, a, alpha_b, Cp, Cn, eps,
             si, /* shrinking */ 0);
}

int Solver_Parallel_SMO_NU::select_working_set(int *work_set,
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
//    int min1 = -1;
//    int min2 = -1;
//    int max1 = -1;
//    int max2 = -1;
    for(int t=0; t<l; ++t)
    {
        if(y[t] == +1)
        {
            if(!is_upper_bound(t))
            {
                if(G[t] < Gmin1)
                {
                    Gmin1 = G[t];
//                    min1 = t;
                }
            }
            if(!is_lower_bound(t))
            {
                if(G[t] > Gmax1)
                {
                    Gmax1 = G[t];
//                    max1 = t;
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
//                    min2 = t;
                }
            }
            if(!is_lower_bound(t))
            {
                if(G[t] > Gmax2)
                {
                    Gmax2 = G[t];
//                    max2 = t;
                }
            }
        }
    }
    //Gmin1 is the smallest G[t] value for which y[t] == +1
    //Gmax1 is the biggest G[t] value for which y[t] == +1
    //Gmin2 is the smallest G[t] value for which y[t] != +1
    //Gmax2 is the biggest G[t] value for which y[t] != +1
    //All excluding boundaries.
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
    // TODO select_working_set(...) is only called if iter > 0 => nselect will always be q?
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
            ++work_count[t]; //TODO update old_idx?
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
        if(eps > 1e-10)
            eps /= 100;
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

void Solver_Parallel_SMO::solve_inner()
{
    Solver s;
    s.Solve(n, SVQ_No_Cache(Q_bb, QD_b, n), c, a, alpha_b, Cp, Cn, eps,
            si, /* shrinking */ 0);
}

int Solver_Parallel_SMO::select_working_set(int *work_set, int *not_work_set)
{
    // printf("selecting working set...");
    // reset work status
    n = n_old;
    int *old_work_set = new int[n];
    for(int i=0; i<n; ++i)
        old_work_set[i] = work_set[i];
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
    //  printf("Gmax1 + Gmax2 = %g < %g\n", Gmax1+Gmax2,eps);
    info(" %g\n", Gmax1+Gmax2);
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
    //  double sort_time = MPI_Wtime();
    quick_sort(yG, pidx, 0, l-1);
    // printf("sort_time = %g\n", MPI_Wtime() - sort_time);
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
    // TODO select_working_set(...) is only called if iter > 0 => nselect will always be q?
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
            {
                old_idx[i] = -1;
                ++nnew;
            }
            else
            {
                //TODO work_set appears to be sorted - use binary search?
                for(int tt=0; tt<n_old; ++tt)
                {
                    if(old_work_set[tt] == t)
                    {
                        old_idx[i] = tt;
                        break;
                    }
                }
            }
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
    //  printf("nnew = %d\n", nnew);
    int kin = nnew;
    nnew = nnew % 2 == 0 ? nnew : nnew-1;
    int L = n/10 % 2 == 0 ? n/10 : (n/10)-1;
    q = min(q, max( max( 10, L ), nnew ) );
    //  printf("q = %d\n", q);
    //  printf("n = %d\n",n);
    if(kin == 0)
    {
        // 1st: Increase precision of solver.
        if(eps > 1e-10)
            eps /= 100;
        else
        {
            info("Error: Unable to select a suitable working set!!!\n");
            return 1;
        }
    }
    // Clean up
    delete[] yG;
    delete[] pidx;
    delete[] old_work_set;
    //  printf("done.\n");
    return 0;
}

void Solver_Parallel_SMO::init_working_set(int *work_set, int *not_work_set)
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

void Solver_Parallel_SMO::Solve(int l, const QMatrix& Q, const double *b_,
                                const schar *y_, double *alpha_, double Cp,
                                double Cn, double eps, SolutionInfo* si,
                                int shrinking)
{
    double total_time = MPI_Wtime();
    double problem_setup_time = 0;
    double inner_solver_time = 0;
    double gradient_updating_time = 0;
    double time = 0;

    // Initialization
    this->l = l;
    this->active_size = l;
    this->Q = &Q;
    this->si = si;
    QD = Q.get_QD();
    this->si = si;
    clone(b,b_,l);
    clone(y,y_,l);
    clone(alpha,alpha_,l);
    this->Cp = Cp;
    this->Cn = Cn;
    this->eps = eps;
    this->lmn = l - n;
    this->G_n = new double[lmn];
    int *work_set = new int[n];
    int *not_work_set = new int[l];
    old_idx = new int[n];
    memset(old_idx, -1, sizeof(int)*n);
    double *delta_alpha = new double[n];

    // Setup alpha, work and parallel cache status
    {
        alpha_status = new char[l];
        work_status = new char[l];
        work_count = new int[l];
        p_cache_status = new char[size*l];
        idx_cached = new int[n];
        idx_not_cached = new int[n];
        nz = new int[n];

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
    {
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
//     this->local_l = l_up_loc - l_low_loc;
//     this->local_n = n_up_loc - n_low_loc;
//     this->local_lmn = lmn_up_loc - lmn_low_loc;
    }

    // Setup gradient
    {
        info("Initializing gradient...");
        info_flush();
        G = new double[l];
        double *G_send = new double[l];
        double *G_recv = new double[l];
        //TODO memcpy & memset or calloc
        for(int i=0; i<l; ++i)
        {
            G[i] = b[i];
            G_send[i] = 0;
        }
        // Determine even distribution of work w.r.t.
        // variables at lower bound
        int k=0;
        int count_not_lower=0;
        int *idx_not_lower = new int[l];
        for(int i=0; i<l; ++i)
        {
            if(!is_lower_bound(i))
            {
                if(rank == k)
                {
                    // We have to compute it
                    idx_not_lower[count_not_lower]=i;
                    ++count_not_lower;
                }
                k = k == size-1 ? 0 : k+1;
            }
        }
        // Compute local portion of gradient
        for(int i=0; i<count_not_lower; ++i)
        {
            const Qfloat *Q_i = Q.get_Q(idx_not_lower[i],l);
            double alpha_i = alpha[idx_not_lower[i]];
            for(int j=0; j<l; ++j)
                G_send[j] += alpha_i * Q_i[j];
        }
        delete[] idx_not_lower;
        // Get contributions from other processors
        /*for(int k=0; k<size; ++k)
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
        }*/
        MPI_Allreduce(G_send,G,l,MPIfloat,MPI_SUM,comm);
        delete[] G_recv;
        delete[] G_send;
        info("done.\n");
        info_flush();
        ierr = MPI_Barrier(comm);
        CheckError(ierr);
    }
    // Allocate space for local subproblem
    Q_bb = new Qfloat[n*n];
    QD_b = new Qfloat[n];
    alpha_b = new double[n];
    c = new double[n];
    a = new schar[n];

//   time = clock();
//   double tmp = 0;
//   for(int i=0; i<100; ++i)
//     {
//       for(int j=0; j<l; ++j)
//         {
//           tmp = Q.get_non_cached(i,j);
//         }
//     }
//   time = clock() - time;
//   printf("Computation time for 100 kernel rows = %.2lf\n", (double)time/CLOCKS_PER_SEC);

    if(rank == 0)
    {
        info("  it  | setup time | solver it | solver time | gradient time ");
        info("| kkt violation\n");
    }

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
                time = MPI_Wtime();
                // info("Starting working set selection\n"); info_flush();
                status = select_working_set(work_set, not_work_set);
                // info("select ws time = %.2f\n", MPI_Wtime() - time);
            }
            else
            {
                // info("Starting init ws\n"); info_flush();
                init_working_set(work_set, not_work_set);
            }
        }

        // Send status to other processors.
        ierr = MPI_Bcast(&status, 1, MPI_INT, 0, comm);
        // Now check for optimality
        if(status != 0)
            break;
        // Send new eps, as select_working_set might have
        // changed it.
        ierr = MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, comm);
        // Send new working set size and working set to other processors.
        ierr = MPI_Bcast(&n, 1, MPI_INT, 0, comm);
        ierr = MPI_Bcast(work_set, n, MPI_INT, 0, comm);
        lmn = l - n;
        ierr = MPI_Bcast(not_work_set, lmn, MPI_INT, 0, comm);

        // Recompute ranges, as n and lmn might have changed
        setup_range(n_low, n_up, n);
        this->n_low_loc = n_low[rank];
        this->n_up_loc = n_up[rank];
        setup_range(lmn_low, lmn_up, lmn);
        //TODO: Only used in LOQO Solver?
        this->lmn_low_loc = lmn_low[rank];
        this->lmn_up_loc = lmn_up[rank];
//       this->local_n = n_up_loc - n_low_loc;
//       this->local_lmn = lmn_up_loc - lmn_low_loc;

        ++iter;
        // Setup subproblem
        time = MPI_Wtime();
        for(int i=0; i<n; ++i)
        {
            c[i] = G[work_set[i]];
            alpha_b[i] = alpha[work_set[i]];
            a[i] = y[work_set[i]];
        }
        //      info("Setting up Q_bb..."); info_flush();
        //TODO: work is very unbalanced because every process only calculates half the matrix.
        for(int i=n_low_loc; i<n_up_loc; ++i)
        {
// 	  const Qfloat *Q_i = Q.get_Q_subset(work_set[i],work_set,n);
            if(Q.is_cached(work_set[i]))
            {
                const Qfloat *Q_i = Q.get_Q_subset(work_set[i],work_set,n);
                for(int j=0; j<=i; ++j)
                {
                    Q_bb[i*n+j] = Q_i[work_set[j]];
                }
            }
            else if(old_idx[i] == -1) //TODO old_idx is only written for rank = 0 and not written for Parallel_Solver_NU?
            {
                for(int j=0; j<=i; ++j)
                {
                    // 	      Q_bb[i*n+j] = Q_i[work_set[j]];
                    Q_bb[i*n+j] = Q.get_non_cached(work_set[i],work_set[j]);
                }
            }
            else // => old_idx[i] != -1 => we know an old index.
            {
                for(int j=0; j<i; ++j)
                {
                    if(old_idx[j] == -1)
                        Q_bb[i*n+j] = Q.get_non_cached(work_set[i],work_set[j]);
                    else
                        Q_bb[i*n+j] = Q_bb[old_idx[j]*n+old_idx[i]];
                }
                Q_bb[i*n+i] = Q.get_non_cached(work_set[i],work_set[i]);
            }
        }
        // Synchronize Q_bb
        ierr = MPI_Barrier(comm);
        CheckError(ierr);
        int num_elements = 0;
        //TODO Allgather/Alltoall?
        //Do not need to send the full row, because only j<i has been changed?
        //Every Process sends his part of Q_bb to all other processes
        for(int k=0; k<size; ++k)
        {
            ierr = MPI_Bcast(&Q_bb[num_elements], (n_up[k]-n_low[k])*n,
                             MPI_FLOAT, k, comm);
            CheckError(ierr);
            num_elements += (n_up[k]-n_low[k])*n;
        }
        //TODO: Do all processes need to create the full Q_bb?
        // Complete symmetric Q
        for(int i=0; i<n; ++i)
        {
            for(int j=0; j<i; ++j)
                Q_bb[j*n+i] = Q_bb[i*n+j];
        }
        for(int i=0; i<n; ++i)
        {
            for(int j=0; j<n; ++j)
            {
                if(alpha[work_set[j]] > TOL_ZERO)
                    c[i] -= Q_bb[i*n+j]*alpha[work_set[j]]; //TODO c is only used on rank == 0?
            }
            QD_b[i] = Q_bb[i*n+i];
        }
        //info("done.\n"); info_flush();
        time = MPI_Wtime() - time;
        problem_setup_time += time;
        if(rank == 0)
        {
            info("%5d | %10.2f |",iter,time);
            info_flush();
            time = MPI_Wtime();
            // Call SMO inner solver
            solve_inner();
            time = MPI_Wtime() - time;
            info(" %11.2f |", time);
            info_flush();
            inner_solver_time += time;
        }
        // Send alpha_b to other processors
        ierr = MPI_Bcast(alpha_b, n, MPI_DOUBLE, 0, comm);
        CheckError(ierr);

        // Update gradient.
        time = MPI_Wtime();
        for(int i=0; i<n; ++i)
        {
            delta_alpha[i] = alpha_b[i] - alpha[work_set[i]];
            if(fabs(delta_alpha[i]) > TOL_ZERO)
                nz[i] = 1;
            else
                nz[i] = 0;
        }
        //      info("Updating G_b..."); info_flush();
        // Compute G_b
        if(rank == 0)
        {
            // Only first processor does updating, since
            // it has the whole Q_bb
            for(int i=0; i<n; ++i)
                for(int j=0; j<n; ++j) // G_b
                {
                    if(nz[j])
                        G[work_set[i]] += Q_bb[n*i+j]*delta_alpha[j];
                }
        }
        //      info("done.\n"); info_flush();

        determine_cached(work_set);

        //   info("count_cached[%d] = %d, count_not_cached[%d] = %d\n", rank,
//       	   count_cached, rank, count_not_cached); info_flush();
//       MPI_Barrier(comm);

        //      printf("count_cached = %d\n", count_cached);
        // info("Updating G_n..."); info_flush();
        // Compute G_n
        for(int j=0; j<lmn; ++j)
            G_n[j] = 0;
        // First update the cached part...
        for(int i=0; i<count_cached; ++i)
        {
            const Qfloat *Q_i = Q.get_Q_subset(work_set[idx_cached[i]],
                                               not_work_set,lmn);
            for(int j=0; j<lmn; ++j)
                G_n[j] += Q_i[not_work_set[j]] * delta_alpha[idx_cached[i]];
        }
        // ...now update the non-cached part
        for(int i=0; i<count_not_cached; ++i)
        {
            const Qfloat *Q_i = Q.get_Q_subset(work_set[idx_not_cached[i]],
                                               not_work_set,lmn);
            for(int j=0; j<lmn; ++j)
                G_n[j] += Q_i[not_work_set[j]] * delta_alpha[idx_not_cached[i]];
        }
        //      info("done.\n"); info_flush();

        // Synchronize gradient with other processors
        //      info("Synchronizing gradient..."); info_flush();
        sync_gradient(work_set, not_work_set);
        //      info("done.\n"); info_flush();

        time = MPI_Wtime() - time;
        gradient_updating_time += time;
        if(rank == 0)
            info(" %13.2f |", time);
        info_flush();

        // Update alpha
        for(int i=0; i<n; ++i)
        {
            alpha[work_set[i]] = alpha_b[i];
            update_alpha_status(work_set[i]);
        }
    } // while(1)
    // Calculate rho
    si->rho = calculate_rho(); //TODO: Only needed on rank == 0?

    // Calculate objective value //TODO: Only needed on rank == 0?
    {
        double v = 0;
        int i;
        for(i=0; i<l; i++)
            v += alpha[i] * (G[i] + b[i]);

        si->obj = v/2;
    }

    // Put back the solution
    {
        for(int i=0; i<l; i++)
            alpha_[i] = alpha[i];
    }

    si->upper_bound_p = Cp;
    si->upper_bound_n = Cn;

    total_time = MPI_Wtime() - total_time;
    // print timing statistics
    if(rank == 0)
    {
        info("\n");
        info("Total opt. time = %.2lf\n", total_time);
        info_flush();
        info("Problem setup time = %.2lf (%.2lf%%)\n", problem_setup_time,
             problem_setup_time/total_time*100);
        info_flush();
        info("Inner solver time = %.2lf (%.2lf%%)\n", inner_solver_time,
             inner_solver_time/total_time*100);
        info_flush();
        info("Gradient updating time = %.2lf (%.2lf%%)\n",
             gradient_updating_time,
             gradient_updating_time/total_time*100);
        info_flush();
    }
    MPI_Barrier(comm);
    info("\noptimization finished, #iter = %d\n",iter);

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
    delete[] QD_b;
    delete[] alpha_b;
    delete[] c;
    delete[] a;
    delete[] l_low;
    delete[] l_up;
    delete[] n_low;
    delete[] n_up;
    delete[] lmn_low;
    delete[] lmn_up;
    delete[] G_n;
    delete[] p_cache_status;
    delete[] idx_cached;
    delete[] idx_not_cached;
    delete[] nz;
}

void Solver_Parallel_SMO::determine_cached(int *work_set)
{
    count_cached = 0;
    count_not_cached = 0;
    // Update local part of the parallel cache status
    for(int i=0; i<l; ++i)
    {
        if(Q->is_cached(i))
            p_cache_status[rank*l + i] = CACHED;
        else
            p_cache_status[rank*l + i] = NOT_CACHED;
    }
    // Synchronize parallel cache status
    //TODO Use MPI-AllGather/AlltoAll?
    /*for(int k=0; k<size; ++k)
    {
        ierr = MPI_Bcast(&p_cache_status[k*l], l, MPI_CHAR, k, comm);
        CheckError(ierr);
    }*/
    //XXX: Check if sendbuf can point into recvbuf - if not we need to create a copy.
    MPI_Allgather(&p_cache_status[rank*l],l,MPI_CHAR,p_cache_status,l,MPI_CHAR,comm);

    // Smart parallel cache handling
    int next_k = 0;
    bool found = false;
    int next_not_cached = 0;
    for(int i=0; i<n; ++i)
    {
        if(nz[i])
        {
            for(int k=next_k; !found && k<size; ++k)
            {
                if(p_cache_status[k*l + work_set[i]] == CACHED)
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
                if(p_cache_status[k*l + work_set[i]] == CACHED)
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
}

void Solver_Parallel_SMO::setup_range(int *range_low, int *range_up,
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

void Solver_Parallel_SMO::sync_gradient(int *work_set, int *not_work_set)
{
    // Synchronize G_b
    double *G_buf = new double[l];
    //TODO not needed copy?
    if(rank == 0)
    {
        for(int i=0; i<n; ++i)
            G_buf[i] = G[work_set[i]];
    }
    ierr = MPI_Bcast(G_buf, n, MPI_DOUBLE, 0, comm);
    CheckError(ierr);
    if(rank != 0)
    {
        for(int i=0; i<n; ++i)
            G[work_set[i]] = G_buf[i];
    }

    // Synchronize G_n
    //TODO Can i use Allreduce or something like that?
    /*for(int i=0; i<size; ++i)
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
    }*/
    MPI_Allreduce(G_n,G_buf,lmn,MPI_DOUBLE,MPI_SUM,comm);
    for(int j=0; j<lmn; ++j)
        G[not_work_set[j]] += G_buf[j];
    delete[] G_buf;
}
